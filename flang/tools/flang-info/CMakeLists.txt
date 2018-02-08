set( LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Option
  Support
  )

add_flang_executable(flang-info
  FlangInfo.cpp
  )

target_link_libraries(flang-info
  flangBasic
  flangSema
  )

install(TARGETS flang-info
  RUNTIME DESTINATION bin)
