
######## f18 ##########

add_executable( f18  
  f18.cc
)
target_link_libraries( f18
  FlangParser
  )

######## test-type ##########

add_executable( test-type  
  test-type.cc
)

target_link_libraries( test-type
  FlangParser
  FlangSemantics
  )


