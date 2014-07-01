""" Utility module to help debug Python scripts

	--------------------------------------------------------------------------
	File: 			utilsDebug.py

	Overview: 		Python module to supply functions to help debug Python
					scripts.
									
	Environment:	OS:			Windows/LINUX/OSX
					IDE: 	    Visual Studio 2013 Plugin Python Tools (PTVS)
					Script:		Python 2.6 x64 

	Gotchas:		None.

	Copyright:		None.
	--------------------------------------------------------------------------
	
"""

# Python modules:

# Third party modules:

# In-house modules:

# Instantiations:

#-----------------------------------------------------------------------------
# Details:	Class to implement simple stack function trace. Instantiation the
#			class as the first function you want to trace. Example:
#				obj = utilsDebug.CDebugFnVerbose( "validate_arguments()" );
# Gotchas:	This class will not work in properly in a multi-threaded 
#			environment.
# Authors:	Illya Rudkin 28/11/2013.
# Changes:	None.
#--
class CDebugFnVerbose:
	# Public static properties:
	bVerboseOn = False;	# True = turn on function tracing, False = turn off.
		
	# Public:
	#++------------------------------------------------------------------------
	# Details:	CDebugFnVerbose constructor.
	# Type:		Method.
	# Args:		vstrFnName	- (R) Text description i.e. a function name.
	# Return:	None.
	# Throws:	None.
	#--
	# CDebugFnVerbose( vstrFnName )
	
	#++------------------------------------------------------------------------
	# Details:	Print out information on the object specified.
	# Type:		Method.
	# Args:		vstrText	- (R) Some helper text description.
	#			vObject		- (R) Some Python type object.
	# Return:	None.
	# Throws:	None.
	#--
	def dump_object( self, vstrText, vObject ):
		if CDebugFnVerbose.bVerboseOn == False:
			return;
		print "%d%s> Dp: %s" % (CDebugFnVerbose.__nLevel, self.__get_dots(), 
								vstrText),;
		print vObject;
	
	#++------------------------------------------------------------------------
	# Details:	Print out some progress text given by the client.
	# Type:		Method.
	# Args:		vstrText	- (R) Some helper text description.
	# Return:	None.
	# Throws:	None.
	#--
	def dump_text( self, vstrText ):
		if CDebugFnVerbose.bVerboseOn == False:
			return;
		print "%d%s> Dp: %s" % (CDebugFnVerbose.__nLevel, self.__get_dots(),
								vstrText);
				
	# Private methods:
	def __init__( self, vstrFnName ):
		self.__indent_out( vstrFnName );
		
	# Class destructor 
	def __del__( self ):
		self.__indent_back();
		
	#++------------------------------------------------------------------------
	# Details:	Build an indentation string of dots based on the __nLevel.
	# Type:		Method.
	# Args:		None.
	# Return:	Str -	variable length string.
	# Throws:	None.
	#--
	def __get_dots( self ):
		return "".join( "." for i in range( 0, CDebugFnVerbose.__nLevel ) );
	
	#++------------------------------------------------------------------------
	# Details:	Build and print out debug verbosity text indicating the function
	#			just exited from.
	# Type:		Method.
	# Args:		None.
	# Return:	None.
	# Throws:	None.
	#--
	def __indent_back( self ):
		if CDebugFnVerbose.bVerboseOn:
			print "%d%s< fn: %s" % (CDebugFnVerbose.__nLevel, self.__get_dots(),
									self.__strFnName);
		CDebugFnVerbose.__nLevel -= 1;

	#++------------------------------------------------------------------------
	# Details:	Build and print out debug verbosity text indicating the function
	#			just entered.
	# Type:		Method.
	# Args:		vstrFnName - (R) Name of the function entered.
	# Return:	None.
	# Throws:	None.
	#--
	def __indent_out( self, vstrFnName ):
		CDebugFnVerbose.__nLevel += 1;
		self.__strFnName = vstrFnName;
		if CDebugFnVerbose.bVerboseOn:
			print "%d%s> fn: %s" % ( CDebugFnVerbose.__nLevel, self.__get_dots(), 
									 self.__strFnName);

	# Private statics attributes:
	__nLevel = 0;	# Indentation level counter
	
	# Private attributes:
	__strFnName = "";
	
	
	