// RUN: %clang -E -F%S %s 2>&1 | grep "published framework headers should always #import headers within the framework with framework paths"

// rdar://7520940
#include <foo/foo.h>

