// RUN: clang-cc -emit-llvm-only -verify %s

struct XPTParamDescriptor {};
struct nsXPTParamInfo {
  nsXPTParamInfo(const XPTParamDescriptor& desc);
};
void a(XPTParamDescriptor *params) {
  const nsXPTParamInfo& paramInfo = params[0];
}
