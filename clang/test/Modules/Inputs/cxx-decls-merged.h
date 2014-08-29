extern int mergeUsedFlag;

typedef struct {
  int n;
  int m;
} NameForLinkage;
extern NameForLinkage name_for_linkage;

struct HasVirtualFunctions {
  virtual void f();
};
struct OverridesVirtualFunctions : HasVirtualFunctions {
  void f();
};
extern OverridesVirtualFunctions overrides_virtual_functions;
extern "C" void ExternCFunction();
