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
  std::filesystem::path policy_path = pack_path / std::string(param_("policy_path"));
  std::filesystem::path mean_path = pack_path / std::string(param_("mean_path"));
  std::filesystem::path var_path = pack_path / std::string(param_("var_path"));

  module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load(policy_path.string()));
  RSFATAL_IF(module_ == nullptr, "module is not loaded");

  std::string in_line;
  std::ifstream obsMean_file(mean_path.string());
  std::ifstream obsVariance_file(var_path.string());
  obs_.setZero(raibotController_.getObDim());
  obsMean_.setZero(raibotController_.getObDim());
  obsVariance_.setZero(raibotController_.getObDim());

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

  obsMean_file.close();
  obsVariance_file.close();
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
  obs_ = raibotController_.getObservation().cast<float>();
  for (int i = 0; i < obs_.size(); ++i) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
    if (obs_(i) > 10) { obs_(i) = 10.0; }
    else if (obs_(i) < -10) { obs_(i) = -10.0; }
  }

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(eigenVectorToTorchTensor(obs_));
  Eigen::VectorXf action = torchTensorToEigenVector(module_->forward(inputs).toTensor());
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