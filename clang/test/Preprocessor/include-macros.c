// RUN: clang-cc -E -Dtest=FOO -imacros %S/pr2086.h %s | grep 'HERE: test'

// This should not be expanded into FOO because pr2086.h undefs 'test'.
HERE: test
