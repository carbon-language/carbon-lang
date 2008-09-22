# NOTE: The tools are organized into five groups of four consisting of one
# large and three small executables. This is done to minimize memory load
# in parallel builds.  Please retain this ordering.

if( NOT MSVC )
  add_subdirectory(llvm-config)
endif( NOT MSVC )

add_subdirectory(opt)
add_subdirectory(llvm-as)
add_subdirectory(llvm-dis)

add_subdirectory(llc)
add_subdirectory(llvm-ranlib)
add_subdirectory(llvm-ar)
add_subdirectory(llvm-nm)

add_subdirectory(llvm-ld)
add_subdirectory(llvm-prof)
add_subdirectory(llvm-link)
add_subdirectory(lli)

# gccas and gccld are deprecated:
# add_subdirectory(gccas)
# add_subdirectory(gccld)
add_subdirectory(llvm-extract)
add_subdirectory(llvm-db)

add_subdirectory(bugpoint)
add_subdirectory(llvm-bcanalyzer)
add_subdirectory(llvm-stub)
add_subdirectory(llvmc2)
