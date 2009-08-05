PROJECT( tests )

SET( CompilerRT_LIBRARY CompilerRT )

# create test library
# add_library(${CompilerRT_LIBRARY} STATIC support.c cmdline.c)
TARGET_LINK_LIBRARIES( ${CompilerRT_LIBRARY} )

SET( TEST_TARGET_LIBRARIES ${CompilerRT_LIBRARY} )

# create tests
# MACRO_ADD_CHECK_TEST( foo foo.c ${TEST_TARGET_LIBRARIES} )

#ADD_SUBDIRECTORY( timing )
#ADD_SUBDIRECTORY( Unit )
