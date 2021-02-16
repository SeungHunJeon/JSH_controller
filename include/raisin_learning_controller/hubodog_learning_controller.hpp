//
// Created by youmdonghoon on 21. 2. 9..
//

#ifndef RAISIN_LEARNING_CONTROLLER_SRC_RAISIN_LEARNING_CONTROLLER_HUBODOG_LEARNING_CONTROLLER_HPP_
#define RAISIN_LEARNING_CONTROLLER_SRC_RAISIN_LEARNING_CONTROLLER_HUBODOG_LEARNING_CONTROLLER_HPP_

#include "raisin_learning_controller/HubodogController.hpp"
#include "raisin_controller/controller.hpp"
#include "torch/script.h"

namespace raisin {

namespace controller {

class HubodogLearningController : public Controller {

 public:
  bool create(raisim::World* world) final;
  bool init(raisim::World* world) final;
  Eigen::VectorXf obsScalingAndGetAction();
  torch::Tensor eigenVectorToTorchTensor(const Eigen::VectorXf& e);
  Eigen::VectorXf torchTensorToEigenVector(const torch::Tensor& t);
  bool advance(raisim::World* world) final;
  bool reset(raisim::World* world) final;
  bool terminate(raisim::World* world) final;
  bool stop(raisim::World* world) final;

 private:
  raisim::HubodogController hubodogController_;
  Eigen::VectorXf obs_, obsMean_, obsVariance_;
  int clk_ = 0;
  double control_dt_; communication_dt_;
  std::unique_ptr<torch::jit::script::Module> module_;

};
}
}

#endif //RAISIN_LEARNING_CONTROLLER_SRC_RAISIN_LEARNING_CONTROLLER_HUBODOG_LEARNING_CONTROLLER_HPP_
