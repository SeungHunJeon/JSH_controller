//
// Created by youmdonghoon on 21. 2. 9..
//
#include <filesystem>
#include "ament_index_cpp/get_package_prefix.hpp"
#include "raisin_learning_controller/hubodog_learning_controller.hpp"

namespace raisin
{

namespace controller
{

using std::placeholders::_1;
using std::placeholders::_2;

HubodogLearningController::HubodogLearningController()
: Controller("raisin_learning_controller"),
  param_(parameter::ParameterContainer::getRoot()["HubodogLearningController"])
{
  param_.loadFromPackageParameterFile("raisin_learning_controller");

  serviceSetCommand_ = this->create_service<raisin_interfaces::srv::Vector3>(
    "raisin_learning_controller/set_command", std::bind(&HubodogLearningController::setCommand, this, _1, _2));
}

bool HubodogLearningController::create(raisim::World * world)
{
  control_dt_ = 0.02;
  communication_dt_ = 0.002;
  hubodogController_.create(world);

  std::filesystem::path pack_path(ament_index_cpp::get_package_prefix("raisin_learning_controller"));
  std::filesystem::path policy_path = pack_path / std::string(param_("policy_path"));
  std::filesystem::path mean_path = pack_path / std::string(param_("mean_path"));
  std::filesystem::path var_path = pack_path / std::string(param_("var_path"));

  module_ =
    std::make_unique<torch::jit::script::Module>(torch::jit::load(policy_path.string()));
  RSFATAL_IF(module_ == nullptr, "module is not loaded")
  std::string in_line;
  std::ifstream obsMean_file(mean_path.string());
  std::ifstream obsVariance_file(var_path.string());
  obs_.setZero(hubodogController_.getObDim());
  obsMean_.setZero(hubodogController_.getObDim());
  obsVariance_.setZero(hubodogController_.getObDim());

  if (obsMean_file.is_open()) {
    for (int i = 0; i < obsMean_.size(); i++) {
      std::getline(obsMean_file, in_line);
      obsMean_(i) = std::stod(in_line);
    }
  }
  if (obsVariance_file.is_open()) {
    for (int i = 0; i < obsVariance_.size(); i++) {
      std::getline(obsVariance_file, in_line);
      obsVariance_(i) = std::stod(in_line);
    }
  }
  obsMean_file.close();
  obsVariance_file.close();
  return true;
}

bool HubodogLearningController::init(raisim::World * world)
{
  hubodogController_.init(world);
  return true;
}

torch::Tensor HubodogLearningController::eigenVectorToTorchTensor(const Eigen::VectorXf & e)
{
  auto t = torch::empty({1, e.size()});
  Eigen::Map<Eigen::VectorXf> ef(t.data_ptr<float>(), t.size(1), t.size(0));
  ef = e.cast<float>();
  t.requires_grad_(false);
  return t;
}

Eigen::VectorXf HubodogLearningController::torchTensorToEigenVector(const torch::Tensor & t)
{
  Eigen::Map<Eigen::VectorXf> e(t.data_ptr<float>(), t.size(1), t.size(0));
  return e;
}

Eigen::VectorXf HubodogLearningController::obsScalingAndGetAction()
{
  obs_ = hubodogController_.getObservation().cast<float>();
  for (int i = 0; i < obs_.size(); i++) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
    if (obs_(i) > 10) {
      obs_(i) = 10.0;
    }
    if (obs_(i) < -10) {
      obs_(i) = -10.0;
    }
  }
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(eigenVectorToTorchTensor(obs_));
  Eigen::VectorXf action = torchTensorToEigenVector(module_->forward(inputs).toTensor());
  return action;
}

bool HubodogLearningController::advance(raisim::World * world)
{
  if (clk_ % int(control_dt_ / communication_dt_ + 1e-10) == 0) { /// 50Hz
    hubodogController_.updateObservation(world);
    hubodogController_.advance(world, obsScalingAndGetAction());
    hubodogController_.updatePreviousAction();
  }
  if (clk_ % (int(control_dt_ / communication_dt_ + 1e-10) / 2 == 0) { /// 100Hz
    hubodogController_.updateHistory();
  }

  clk_++;
  return true;
}

bool HubodogLearningController::reset(raisim::World * world)
{
  hubodogController_.reset(world);
  clk_ = 0;
  return true;
}

bool HubodogLearningController::terminate(raisim::World * world)
{
  return true;
}

bool HubodogLearningController::stop(raisim::World * world)
{
  return true;
}

extern "C" Controller * create()
{
  return new HubodogLearningController;
}

extern "C" void destroy(Controller * p)
{
  delete p;
}

void HubodogLearningController::setCommand(
  const std::shared_ptr<raisin_interfaces::srv::Vector3::Request> request,
  std::shared_ptr<raisin_interfaces::srv::Vector3::Response> response)
try {
  Eigen::Vector3f command;
  command << request->x, request->y, request->z;
  hubodogController_.setCommand(command);
  response->success = true;
} catch (const std::exception& e) {
  response->success = false;
  response->message = e.what();
}

}

}
