// RUN: env MACOSX_DEPLOYMENT_TARGET=10.1 clang -E %s

#if __ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__ != 1010
#error Invalid version
#endif

