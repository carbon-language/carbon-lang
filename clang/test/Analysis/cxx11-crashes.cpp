// RUN: %clang_cc1 -analyze -analyzer-checker=core -std=c++11 -verify %s

// radar://11485149, PR12871
class PlotPoint {
  bool valid;
};

PlotPoint limitedFit () {
  PlotPoint fit0;
  fit0 = limitedFit ();
  return fit0;
}

// radar://11487541, NamespaceAlias
namespace boost {namespace filesystem3 {
class path {
public:
 path(){}
};

}}
namespace boost
{
  namespace filesystem
  {
    using filesystem3::path;
  }
}

void radar11487541() {
  namespace fs = boost::filesystem;
  fs::path p;
}
