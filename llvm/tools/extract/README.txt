I checked in a new tool, primarily useful for debugging.  Given a module 
and a function name, it extracts just the specified function from the 
module, with a minimum of related cruft (global variables, function 
prototypes, etc).

This is useful because often something will die (for example SCCP 
miscompiles one function of a large benchmark), and so you want to just 
cut the testcase down to the one function that is being a problem.  In 
this case, 'extract' eliminates all of the extraneous global variables, 
type information, and functions that aren't necessary, giving you 
something simpler.

This is just an FYI, because I've found it useful and thought you guys 
might as well.

