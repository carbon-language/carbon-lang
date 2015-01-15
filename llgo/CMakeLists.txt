include(ExternalProject)
include(ProcessorCount)

# Provide a config.h which exposes build system information.
configure_file(
  cmd/gllgo/config.h.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmd/gllgo/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/cmd/gllgo)

llvm_add_go_executable(llgo llvm.org/llgo/cmd/gllgo ALL DEPENDS
  build/context.go
  cmd/gllgo/gllgo.go
  debug/debug.go
  driver/parser.go
  irgen/annotations.go
  irgen/attribute.go
  irgen/builtins.go
  irgen/cabi.go
  irgen/call.go
  irgen/channels.go
  irgen/closures.go
  irgen/compiler.go
  irgen/errors.go
  irgen/indirect.go
  irgen/interfaces.go
  irgen/maps.go
  irgen/predicates.go
  irgen/println.go
  irgen/runtime.go
  irgen/slice.go
  irgen/ssa.go
  irgen/strings.go
  irgen/targets.go
  irgen/typemap.go
  irgen/types.go
  irgen/utils.go
  irgen/value.go
  irgen/version.go
  ssaopt/esc.go
)

llvm_add_go_executable(llgo-stage2 llvm.org/llgo/cmd/gllgo
  DEPENDS libgo ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
  GOFLAGS "cc=${CMAKE_BINARY_DIR}/bin/clang"
          "cxx=${CMAKE_BINARY_DIR}/bin/clang++"
          "llgo=${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}"
)

llvm_add_go_executable(llgo-stage3 llvm.org/llgo/cmd/gllgo
  DEPENDS libgo ${CMAKE_BINARY_DIR}/bin/llgo-stage2${CMAKE_EXECUTABLE_SUFFIX}
  GOFLAGS "cc=${CMAKE_BINARY_DIR}/bin/clang"
          "cxx=${CMAKE_BINARY_DIR}/bin/clang++"
          "llgo=${CMAKE_BINARY_DIR}/bin/llgo-stage2${CMAKE_EXECUTABLE_SUFFIX}"
)

llvm_add_go_executable(cc-wrapper llvm.org/llgo/cmd/cc-wrapper DEPENDS
  cmd/cc-wrapper/main.go
)

llvm_add_go_executable(llgoi llvm.org/llgo/cmd/llgoi
  DEPENDS libgo ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
          cmd/llgoi/isatty_posix.go
          cmd/llgoi/llgoi.go
  GOFLAGS "cc=${CMAKE_BINARY_DIR}/bin/clang"
          "cxx=${CMAKE_BINARY_DIR}/bin/clang++"
          "llgo=${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}"
)

install(FILES ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
              ${CMAKE_BINARY_DIR}/bin/llgoi${CMAKE_EXECUTABLE_SUFFIX}
        DESTINATION bin
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)

function(add_clobber_steps name)
  ExternalProject_Add_Step(${name} force-reconfigure
    DEPENDERS configure
    ALWAYS 1
    )

  ExternalProject_Add_Step(${name} clobber
    COMMAND ${CMAKE_COMMAND} -E remove_directory <BINARY_DIR>
    COMMAND ${CMAKE_COMMAND} -E make_directory <BINARY_DIR>
    COMMENT "Clobbering ${name} build directory..."
    DEPENDERS configure
    DEPENDS ${ARGN}
    )
endfunction()

processorcount(PROCESSOR_COUNT)

function(add_libgo_variant suffix cflags gocflags deps exclude_from_all)
  externalproject_add(libbacktrace${suffix}
    DEPENDS clang ${deps}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libbacktrace
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${suffix}/libbacktrace
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --disable-multilib --enable-host-shared "CC=${CMAKE_BINARY_DIR}/bin/clang ${cflags}"
    BUILD_COMMAND make -j${PROCESSOR_COUNT}
    INSTALL_COMMAND ""
    LOG_CONFIGURE 1
    LOG_BUILD 1
  )
  set_property(TARGET libbacktrace${suffix}
               PROPERTY EXCLUDE_FROM_ALL ${exclude_from_all})

  add_clobber_steps(libbacktrace${suffix} clang ${deps})

  externalproject_add(libffi${suffix}
    DEPENDS clang ${deps}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libffi
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${suffix}/libffi
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --disable-multilib "CC=${CMAKE_BINARY_DIR}/bin/clang ${cflags}"
    BUILD_COMMAND make -j${PROCESSOR_COUNT}
    INSTALL_COMMAND ""
    LOG_CONFIGURE 1
    LOG_BUILD 1
  )
  set_property(TARGET libffi${suffix}
               PROPERTY EXCLUDE_FROM_ALL ${exclude_from_all})

  add_clobber_steps(libffi${suffix} clang ${deps})

  externalproject_add(libgo${suffix}
    DEPENDS clang llgo cc-wrapper libbacktrace${suffix} libffi${suffix} ${deps}
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/gofrontend/libgo
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/${suffix}/libgo
    INSTALL_DIR ${CMAKE_BINARY_DIR}
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --disable-multilib --without-libatomic --prefix=<INSTALL_DIR> "CC=env REAL_CC=${CMAKE_BINARY_DIR}/bin/clang@SPACE@${cflags} ${CMAKE_BINARY_DIR}/bin/cc-wrapper" "GOC=${CMAKE_BINARY_DIR}/bin/llgo -no-prefix -fcompilerrt-prefix=${CMAKE_BINARY_DIR} ${gocflags}"
    BUILD_COMMAND make -j${PROCESSOR_COUNT}
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
  )
  set_property(TARGET libgo${suffix}
               PROPERTY EXCLUDE_FROM_ALL ${exclude_from_all})

  add_clobber_steps(libgo${suffix} clang
    ${CMAKE_BINARY_DIR}/bin/llgo${CMAKE_EXECUTABLE_SUFFIX}
    ${CMAKE_BINARY_DIR}/bin/cc-wrapper${CMAKE_EXECUTABLE_SUFFIX})
endfunction()

add_libgo_variant("" "" "" "" FALSE)

if(TARGET asan)
  add_libgo_variant("_asan" "-fsanitize=address" "-fsanitize=address" asan TRUE)
endif()

if(TARGET tsan)
  add_libgo_variant("_tsan" "-fsanitize=thread" "-fsanitize=thread" tsan TRUE)
endif()

if(TARGET msan)
  add_libgo_variant("_msan" "-fsanitize=memory" "-fsanitize=memory" msan TRUE)
endif()

if(TARGET dfsan)
  add_libgo_variant("_dfsan" "-fsanitize=dataflow" "-fsanitize=dataflow" dfsan TRUE)
endif()

set(LLGO_LIBRARY_DIR ${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX})

install(FILES ${LLGO_LIBRARY_DIR}/libgo-llgo.a
              ${LLGO_LIBRARY_DIR}/libgo-llgo.so
              ${LLGO_LIBRARY_DIR}/libgo-llgo.so.6
              ${LLGO_LIBRARY_DIR}/libgo-llgo.so.6.0.0
              ${LLGO_LIBRARY_DIR}/libgobegin-llgo.a
        DESTINATION lib${LLVM_LIBDIR_SUFFIX})

install(DIRECTORY ${LLGO_LIBRARY_DIR}/go
        DESTINATION lib${LLVM_LIBDIR_SUFFIX})

add_custom_target(check-libgo
  COMMAND make -C ${CMAKE_CURRENT_BINARY_DIR}/libgo -j${PROCESSOR_COUNT} check
  DEPENDS libgo
  COMMENT "Running libgo tests")

add_custom_target(check-llgo-bootstrap
  COMMAND strip -R .note.gnu.build-id -o ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage2.stripped
          ${CMAKE_BINARY_DIR}/bin/llgo-stage2${CMAKE_EXECUTABLE_SUFFIX}
  COMMAND strip -R .note.gnu.build-id -o ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage3.stripped
          ${CMAKE_BINARY_DIR}/bin/llgo-stage3${CMAKE_EXECUTABLE_SUFFIX}
  COMMAND cmp ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage2.stripped
          ${CMAKE_CURRENT_BINARY_DIR}/llgo-stage3.stripped
  DEPENDS llgo-stage2 llgo-stage3
  COMMENT "Checking llgo bootstrap")

add_subdirectory(test)
