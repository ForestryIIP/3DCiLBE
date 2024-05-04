#pragma once
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
namespace rangenet {
namespace segmentation {

class Net {
 public:
  typedef std::tuple< u_char, u_char, u_char> color;

  Net(const std::string& model_path);
  virtual ~Net(){};
  virtual std::vector<std::vector<float>> infer(const std::vector<float>& scan, const uint32_t &num_points) = 0;
  std::vector<cv::Vec3f> getPoints(const std::vector<float> &scan, const uint32_t& num_points);
  std::vector<cv::Vec3b> getLabels(const std::vector<std::vector<float> > &semantic_scan, const uint32_t& num_points);
  void verbosity(const bool verbose) { _verbose = verbose; }
  std::vector<int> getLabelMap() { return _lable_map;}
  std::map<uint32_t, color> getColorMap() { return _color_map;}

 protected:

  std::string _model_path;
  bool _verbose;

  int _img_h, _img_w, _img_d;
  std::vector<float> _img_means, _img_stds;
  int32_t _n_classes;
  double _fov_up, _fov_down;

  YAML::Node data_cfg;
  YAML::Node arch_cfg;

  std::vector<int> _lable_map;
  std::map<uint32_t, color> _color_map;
  std::map<uint32_t, color> _argmax_to_rgb;  // for color conversion
};

}
}
