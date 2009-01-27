// RUN: clang -E %s | grep bark

#if 0
blah
#\
else
bark
#endif

