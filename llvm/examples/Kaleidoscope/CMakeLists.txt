add_custom_target(Kaleidoscope)

macro(add_kaleidoscope_chapter name)
  add_dependencies(Kaleidoscope ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_kaleidoscope_chapter name)

add_subdirectory(Chapter2)
add_subdirectory(Chapter3)
add_subdirectory(Chapter4)
add_subdirectory(Chapter5)
add_subdirectory(Chapter6)
add_subdirectory(Chapter7)
add_subdirectory(Chapter8)
