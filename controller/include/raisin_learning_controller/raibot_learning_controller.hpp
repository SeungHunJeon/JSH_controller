//
// Created by suyoung on 8/7/21.
//
#pragma once

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "raisim/World.hpp"
#include "raisin_learning_controller/raibotController.hpp"
#include "raisin_parameter/parameter_container.hpp"
#include "raisin_controller/controller.hpp"
#include "raisin_interfaces/srv/vector3.hpp"
#include "torch/script.h"
#include "helper/neuralNet.hpp"

namespace raisin {

namespace controller {

class raibotLearningController : public Controller {

 public:
  raibotLearningController();
  bool create(raisim::World *world) final;
  bool init(raisim::World *world) final;
  Eigen::VectorXf obsScalingAndGetAction();
  bool advance(raisim::World *world) final;
  bool reset(raisim::World *world) final;
  bool terminate(raisim::World *world) final;
  bool stop(raisim::World *world) final;
  torch::Tensor eigenVectorToTorchTensor(const Eigen::VectorXf &e);
  Eigen::VectorXf torchTensorToEigenVector(const torch::Tensor &t);

 private:
  void setCommand(
      const std::shared_ptr<raisin_interfaces::srv::Vector3::Request> request,
      std::shared_ptr<raisin_interfaces::srv::Vector3::Response> response
      );

  rclcpp::Service<raisin_interfaces::srv::Vector3>::SharedPtr serviceSetCommand_;

  raisim::raibotController raibotController_;
  Eigen::VectorXf obs_, obsMean_, obsVariance_, eoutMean_, eoutVariance_, actor_input_;
  int clk_ = 0;
  double control_dt_, communication_dt_;
  raisim::nn::LSTM_MLP<float, 44, 12, raisim::nn::ActivationType::leaky_relu> actor_;
  raisim::nn::Linear<float, 33, 8, raisim::nn::ActivationType::leaky_relu> estimator_;
//  std::unique_ptr<torch::jit::script::Module> actor_;
//  std::unique_ptr<torch::jit::script::Module> estimator_;

  parameter::ParameterContainer & param_;

};

}

}


