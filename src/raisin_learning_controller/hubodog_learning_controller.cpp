//
// Created by youmdonghoon on 21. 2. 9..
//
#include <raisin_learning_controller/hubodog_learning_controller.hpp>

namespace raisin {

namespace controller {

  bool HubodogLearningController::create(raisim::World* world) {
    control_dt_ = 0.04;
    communication_dt_ = 0.002;
    hubodogController_.create(world);
    module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load("../rsc/policy_20000.pt"));
    RSFATAL_IF(module_ == nullptr, "module is not loaded")
    std::string in_line;
    std::ifstream obsMean_file("../rsc/mean20000.csv");
    std::ifstream obsVariance_file("../rsc/var20000.csv");
    obs_.setZero(hubodogController_.getObDim());
    obsMean_.setZero(hubodogController_.getObDim());
    obsVariance_.setZero(hubodogController_.getObDim());

    if(obsMean_file.is_open()) {
      for(int i = 0; i < obsMean_.size(); i++){
        std::getline(obsMean_file, in_line);
        obsMean_(i) = std::stod(in_line);
      }
    }
    if(obsVariance_file.is_open()) {
      for(int i = 0; i < obsVariance_.size(); i++){
        std::getline(obsVariance_file, in_line);
        obsVariance_(i) = std::stod(in_line);
      }
    }
    obsMean_file.close();
    obsVariance_file.close();
    return true;
  }

  bool HubodogLearningController::init(raisim::World* world) {
    hubodogController_.init(world);
    return true;
  }

  Eigen::VectorXf HubodogLearningController::obsScalingAndGetAction() {
    obs_ = hubodogController_.getObservation().cast<float>();
    for(int i = 0; i < obs_.size(); i++) {
      obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
      if(obs_(i) > 10) obs_(i) = 10.0;
      if(obs_(i) < -10) obs_(i) = -10.0;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(eigenVectorToTorchTensor(obs_));
    return torchTensorToEigenVector(module_->forward(inputs).toTensor());
  }

  torch::Tensor eigenVectorToTorchTensor(const Eigen::VectorXf& e) {
    auto t = torch::empty({1, e.size()});
    Eigen::Map<Eigen::VectorXf> ef(t.data_ptr<float>(),t.size(1),t.size(0));
    ef = e.cast<float>();
    t.requires_grad_(false);
    return t;
  }

  Eigen::VectorXf torchTensorToEigenVector(const torch::Tensor& t) {
    Eigen::Map<Eigen::VectorXf> e(t.data_ptr<float>(), t.size(1), t.size(0));
    return e;
  }

  bool HubodogLearningController::advance(raisim::World* world) {
    if(clk_ % (int(control_dt/communication_dt + 1e-10) / 2) == 0) { /// 50Hz
      hubodogController_.updateObservation(world);
      if(clk_ % int(control_dt/communication_dt + 1e-10) == 0) { /// 25Hz
        hubodogController_.advance(world, obsScalingAndGetAction());
        hubodogController_.updatePreviousAction();
      }
      hubodogController_.updateHistory();
    }
    clk_++;
    return true;
  }

  bool HubodogLearningController::reset(raisim::World* world) {
    hubodogController_.reset(world);
    clk_ = 0;
    return true;
  }

  bool HubodogLearningController::terminate(raisim::World* world) {
    return true;
  }

  bool HubodogLearningController::stop(raisim::World* world) {
    return true;
  }

  extern "C" Controller * create() {
    return new HubodogLearningController;
  }

  extern "C" void destroy(Controller* p) {
    delete p;
  }

}

}
