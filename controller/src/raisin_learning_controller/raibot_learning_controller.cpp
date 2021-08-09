//
// Created by suyoung on 8/7/21.
//

#include <filesystem>
#include "ament_index_cpp/get_package_prefix.hpp"
#include "raisin_learning_controller/raibot_learning_controller.hpp"

namespace raisin {

namespace controller {

using std::placeholders::_1;
using std::placeholders::_2;

raibotLearningController::raibotLearningController()
: Controller("raisin_learning_controller"),
  param_(parameter::ParameterContainer::getRoot()["raibotLearningController"])
{
  param_.loadFromPackageParameterFile("raisin_learning_controller");

  serviceSetCommand_ = this->create_service<raisin_interfaces::srv::Vector3>(
      "raisin_learning_controller/set_command", std::bind(&raibotLearningController::setCommand, this, _1, _2)
      );
}

bool raibotLearningController::create(raisim::World *world) {
  control_dt_ = 0.01;
  communication_dt_ = 0.0025;
  raibotController_.create(world);

  std::filesystem::path pack_path(ament_index_cpp::get_package_prefix("raisin_learning_controller"));
  std::filesystem::path actor_path = pack_path / std::string(param_("actor_path"));
  std::filesystem::path estimator_path = pack_path / std::string(param_("estimator_path"));
  std::filesystem::path obs_mean_path = pack_path / std::string(param_("obs_mean_path"));
  std::filesystem::path obs_var_path = pack_path / std::string(param_("obs_var_path"));
  std::filesystem::path eout_mean_path = pack_path / std::string(param_("eout_mean_path"));
  std::filesystem::path eout_var_path = pack_path / std::string(param_("eout_var_path"));

  actor_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(actor_path.string()));
  estimator_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(estimator_path.string()));
  RSFATAL_IF(actor_ == nullptr, "actor is not loaded");
  RSFATAL_IF(estimator_ == nullptr, "estimator is not loaded");

  std::string in_line;
  std::ifstream obsMean_file(obs_mean_path.string());
  std::ifstream obsVariance_file(obs_var_path.string());
  std::ifstream eoutMean_file(eout_mean_path.string());
  std::ifstream eoutVariance_file(eout_var_path.string());
  obs_.setZero(raibotController_.getObDim());
  obsMean_.setZero(raibotController_.getObDim());
  obsVariance_.setZero(raibotController_.getObDim());
  eoutMean_.setZero(raibotController_.getEstDim());
  eoutVariance_.setZero(raibotController_.getEstDim());
  actor_input_.setZero(raibotController_.getObDim() + raibotController_.getEstDim());

  if (obsMean_file.is_open()) {
    for (int i = 0; i < obsMean_.size(); ++i) {
      std::getline(obsMean_file, in_line);
      obsMean_(i) = std::stof(in_line);
    }
  }

  if (obsVariance_file.is_open()) {
    for (int i = 0; i < obsVariance_.size(); ++i) {
      std::getline(obsVariance_file, in_line);
      obsVariance_(i) = std::stof(in_line);
    }
  }

  if (eoutMean_file.is_open()) {
    for (int i = 0; i < eoutMean_.size(); ++i) {
      std::getline(eoutMean_file, in_line);
      eoutMean_(i) = std::stof(in_line);
    }
  }

  if (eoutVariance_file.is_open()) {
    for (int i = 0; i < eoutVariance_.size(); ++i) {
      std::getline(eoutVariance_file, in_line);
      eoutVariance_(i) = std::stof(in_line);
    }
  }

  obsMean_file.close();
  obsVariance_file.close();
  eoutMean_file.close();
  eoutVariance_file.close();
  return true;
}

bool raibotLearningController::init(raisim::World *world) {
  raibotController_.init(world);
  return true;
}

bool raibotLearningController::advance(raisim::World *world) {
  /// 100Hz controller
  if(clk_ % int(control_dt_ / communication_dt_ + 1e-10) == 0) {
    raibotController_.updateObservation(world);
    raibotController_.advance(world, obsScalingAndGetAction());
  }

  clk_++;
  return true;
}

Eigen::Ref<Eigen::VectorXf> raibotLearningController::obsScalingAndGetAction() {
  /// normalize the obs
  obs_ = raibotController_.getObservation().cast<float>();
  for (int i = 0; i < obs_.size(); ++i) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
    if (obs_(i) > 10) { obs_(i) = 10.0; }
    else if (obs_(i) < -10) { obs_(i) = -10.0; }
  }

  /// forward the obs to the estimator
  std::vector<torch::jit::IValue> e_in;
  e_in.push_back(eigenVectorToTorchTensor(obs_.tail(obs_.size() - 3)));
  Eigen::VectorXf e_out = torchTensorToEigenVector(estimator_->forward(e_in).toTensor());

  /// normalize the output of estimator
  for (int i = 0; i < e_out.size(); ++i) {
    e_out(i) = (e_out(i) - eoutMean_(i)) / std::sqrt(eoutVariance_(i) + 1e-8);
    if (e_out(i) > 10) { e_out(i) = 10.0; }
    else if (e_out(i) < -10) { e_out(i) = -10.0; }
  }

  /// concat obs and e_out and forward to the actor
  actor_input_ << obs_, e_out;
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(eigenVectorToTorchTensor(actor_input_));
  Eigen::VectorXf action = torchTensorToEigenVector(actor_->forward(inputs).toTensor());
  return action;
}

bool raibotLearningController::reset(raisim::World *world) {
  raibotController_.reset(world);
  clk_ = 0;
  return true;
}

bool raibotLearningController::terminate(raisim::World *world) { return true; }

bool raibotLearningController::stop(raisim::World *world) { return true; }

extern "C" Controller * create() {
  return new raibotLearningController;
}

extern "C" void destroy(Controller *p) {
  delete p;
}

torch::Tensor raibotLearningController::eigenVectorToTorchTensor(const Eigen::VectorXf &e) {
  auto t = torch::empty({1, e.size()});
  Eigen::Map<Eigen::VectorXf> ef(t.data_ptr<float>(), t.size(1), t.size(0));
  ef = e.cast<float>();
  t.requires_grad_(false);
  return t;
}

Eigen::VectorXf raibotLearningController::torchTensorToEigenVector(const torch::Tensor &t) {
  Eigen::Map<Eigen::VectorXf> e(t.data_ptr<float>(), t.size(1), t.size(0));
  return e;
}

void raibotLearningController::setCommand(const std::shared_ptr<raisin_interfaces::srv::Vector3::Request> request,
                                          std::shared_ptr<raisin_interfaces::srv::Vector3::Response> response)
try {
  Eigen::Vector3f command;
  command << request->x, request->y, request->z;
  raibotController_.setCommand(command);
  response->success = true;
} catch (const std::exception &e) {
  response->success = false;
  response->message = e.what();
}

}

}