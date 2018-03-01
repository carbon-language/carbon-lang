
######## f18 ##########

add_executable( f18
  f18.cc
)
target_link_libraries( f18
  FortranParser
  )

######## test-type ##########

add_executable( test-type
  test-type.cc
)

target_link_libraries( test-type
  FortranParser
  FlangSemantics
  )
