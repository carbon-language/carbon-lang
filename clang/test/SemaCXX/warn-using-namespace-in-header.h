




// Lots of vertical space to make the error line match up with the line of the
// expected line in the source file.
namespace warn_in_header_in_global_context {}
using namespace warn_in_header_in_global_context;

// While we want to error on the previous using directive, we don't when we are
// inside a namespace
namespace dont_warn_here {
using namespace warn_in_header_in_global_context;
}
