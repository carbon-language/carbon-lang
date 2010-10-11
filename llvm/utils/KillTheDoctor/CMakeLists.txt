add_executable(KillTheDoctor
  KillTheDoctor.cpp
  system_error.cpp
  )

target_link_libraries(KillTheDoctor LLVMSupport LLVMSystem)
