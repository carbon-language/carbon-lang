#ifndef SUPPRESS_ARRAY_WARNINGS_H
#define SUPPRESS_ARRAY_WARNINGS_H

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#pragma GCC diagnostic ignored "-Wmissing-braces"

#endif // SUPPRESS_ARRAY_WARNINGS
