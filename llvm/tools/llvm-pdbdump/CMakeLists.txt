set(MSVC_DIA_SDK_DIR "$ENV{VSINSTALLDIR}DIA SDK")

# Due to a bug in MSVC 2013's installation software, it is possible
# for MSVC 2013 to write the DIA SDK into the Visual Studio 2012
# install directory.  If this happens, the installation is corrupt
# and there's nothing we can do.  It happens with enough frequency
# though that we should handle it.  We do so by simply checking that
# the DIA SDK folder exists.  Should this happen you will need to
# uninstall VS 2012 and then re-install VS 2013.
if (IS_DIRECTORY ${MSVC_DIA_SDK_DIR})
  set(LLVM_LINK_COMPONENTS
    Support
    )

  include_directories(${MSVC_DIA_SDK_DIR}/include)
  if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    link_directories(${MSVC_DIA_SDK_DIR}/lib/amd64)
  else()
    link_directories(${MSVC_DIA_SDK_DIR}/lib)
  endif()

  add_llvm_tool(llvm-pdbdump
    llvm-pdbdump.cpp
    DIAExtras.cpp
    DIASymbol.cpp
    )
  target_link_libraries(llvm-pdbdump diaguids)
endif()
