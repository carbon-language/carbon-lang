#ifndef SUPPORT_DISABLE_MISSING_BRACES_WARNING_H
#define SUPPORT_DISABLE_MISSING_BRACES_WARNING_H

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#pragma GCC diagnostic ignored "-Wmissing-braces"

#endif // SUPPORT_DISABLE_MISSING_BRACES_WARNING_H
