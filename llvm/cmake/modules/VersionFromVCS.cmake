# Adds version control information to the variable VERS. For
# determining the Version Control System used (if any) it inspects the
# existence of certain subdirectories under CMAKE_CURRENT_SOURCE_DIR.

function(add_version_info_from_vcs VERS)
  set(result ${${VERS}})
  if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.svn )
    set(result "${result}svn")
    find_package(Subversion)
    if( Subversion_FOUND )
      subversion_wc_info( ${CMAKE_CURRENT_SOURCE_DIR} Project )
      if( Project_WC_REVISION )
	set(result "${result}-r${Project_WC_REVISION}")
      endif()
    endif()
  elseif( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git )
    set(result "${result}git")
  endif()
  set(${VERS} ${result} PARENT_SCOPE)
endfunction(add_version_info_from_vcs)
