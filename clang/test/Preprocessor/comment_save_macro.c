// RUN: clang -E -C %s | grep '^boo bork bar // zot$' &&
// RUN: clang -E -CC %s | grep -F '^boo bork /* blah*/ bar // zot$' &&
// RUN: clang -E %s | grep '^boo bork bar$'


#define FOO bork // blah
boo FOO bar // zot

