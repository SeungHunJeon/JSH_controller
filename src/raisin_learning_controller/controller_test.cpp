#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <Eigen/Core>

  torch::Tensor eigenVectorToTorchTensor(const Eigen::VectorXf& e) {
    auto t = torch::empty({1, e.size()});
    Eigen::Map<Eigen::VectorXf> ef(t.data_ptr<float>(),t.size(1),t.size(0));
    ef = e.cast<float>();
    t.requires_grad_(false);
    return t;
  }

  Eigen::VectorXf torchTensorToEigenVector(const torch::Tensor& t) {
    Eigen::Map<Eigen::VectorXf> action(t.data_ptr<float>(), t.size(1), t.size(0));
    return action;
}

int main() {
  double control_dt = 0.04, communication_dt = 0.002;
  std::cout<<int(control_dt/communication_dt + 1e-10)<<std::endl;
  Eigen::VectorXf obsMean_, obsVariance_;
  obsMean_.setZero(113); obsVariance_.setZero(113);
  std::string in_line;
  std::ifstream obsMean_file("../rsc/mean20000.csv");
  std::ifstream obsVariance_file("../rsc/var20000.csv");
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

  Eigen::VectorXf obs_;
  obs_.setZero(113);
  obs_ <<
       0.388244,
  0,
  0,
  1,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0.52,
      -1.06,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0,
  0;
  for(int i = 0; i < obs_.size(); i++) {
    obs_(i) = (obs_(i) - obsMean_(i)) / std::sqrt(obsVariance_(i) + 1e-8);
    if(obs_(i) > 10) obs_(i) = 10.0;
    if(obs_(i) < -10) obs_(i) = -10.0;
  }
  std::cout<<obsMean_.transpose()<<std::endl;
  std::cout<<obsVariance_.transpose()<<std::endl;

  torch::Tensor T_ = eigenVectorToTorchTensor(obs_);
  std::cout<<T_<<std::endl;
  std::unique_ptr<torch::jit::script::Module> module_;
  module_ = std::make_unique<torch::jit::script::Module>(torch::jit::load("../rsc/policy_20000.pt"));

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(T_);
  torch::Tensor T2 = module_->forward(inputs).toTensor();
  std::cout<<T2<<std::endl;
  std::cout<<torchTensorToEigenVector(T2)<<std::endl;
return 0;
}