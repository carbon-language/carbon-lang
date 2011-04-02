




// Lots of vertical space to make the error line match up with the line of the
// expected line in the source file.
namespace warn_in_header_in_global_context {}
using namespace warn_in_header_in_global_context;

// While we want to error on the previous using directive, we don't when we are
// inside a namespace
namespace dont_warn_here {
using namespace warn_in_header_in_global_context;
}

// We should warn in toplevel extern contexts.
namespace warn_inside_linkage {}
extern "C++" {
using namespace warn_inside_linkage;
}

// This is really silly, but we should warn on it:
extern "C++" {
extern "C" {
extern "C++" {
using namespace warn_inside_linkage;
}
}
}

// But we shouldn't warn in extern contexts inside namespaces.
namespace dont_warn_here {
extern "C++" {
using namespace warn_in_header_in_global_context;
}
}

// We also shouldn't warn in case of functions.
inline void foo() {
  using namespace warn_in_header_in_global_context;
}


namespace macronamespace {}
#define USING_MACRO using namespace macronamespace;

// |using namespace| through a macro should warn if the instantiation is in a
// header.
USING_MACRO
