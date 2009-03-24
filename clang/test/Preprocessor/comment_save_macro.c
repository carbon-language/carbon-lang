// RUN: clang-cc -E -C %s | grep '^boo bork bar // zot$' &&
// RUN: clang-cc -E -CC %s | grep -F '^boo bork /* blah*/ bar // zot$' &&
// RUN: clang-cc -E %s | grep '^boo bork bar$'


#define FOO bork // blah
boo FOO bar // zot

