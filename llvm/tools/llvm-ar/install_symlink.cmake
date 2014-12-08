# We need to execute this script at installation time because the
# DESTDIR environment variable may be unset at configuration time.
# See PR8397.

if(UNIX)
  set(LINK_OR_COPY create_symlink)
  set(DESTDIR $ENV{DESTDIR})
else()
  set(LINK_OR_COPY copy)
endif()

# CMAKE_EXECUTABLE_SUFFIX is undefined on cmake scripts. See PR9286.
if( WIN32 )
  set(EXECUTABLE_SUFFIX ".exe")
else()
  set(EXECUTABLE_SUFFIX "")
endif()

set(bindir "${DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/")

message("Creating llvm-ranlib")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E ${LINK_OR_COPY} "llvm-ar${EXECUTABLE_SUFFIX}" "llvm-ranlib${EXECUTABLE_SUFFIX}"
  WORKING_DIRECTORY "${bindir}")
