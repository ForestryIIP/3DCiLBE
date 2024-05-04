#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#ifdef TENSORRT_FOUND
#endif

namespace rangenet {
namespace segmentation {


std::unique_ptr<Net> make_net(const std::string& path,
                              const std::string& backend);

}  }
