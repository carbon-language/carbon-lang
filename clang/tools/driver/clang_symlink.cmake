# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

if(UNIX)
  set(CLANGXX_LINK_OR_COPY create_symlink)
  set(CLANGXX_DESTDIR $ENV{DESTDIR})
else()
  set(CLANGXX_LINK_OR_COPY copy)
endif()

set(bindir "${CLANGXX_DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/")
set(clang "clang${CMAKE_EXECUTABLE_SUFFIX}")
set(clangxx "clang++${CMAKE_EXECUTABLE_SUFFIX}")

message("Creating clang++ executable based on ${clang}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E ${CLANGXX_LINK_OR_COPY} "${clang}" "${clangxx}"
  WORKING_DIRECTORY "${bindir}")
