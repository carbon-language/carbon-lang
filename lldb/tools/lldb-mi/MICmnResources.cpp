//===-- MICmnResources.cpp --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmnResources.cpp
//
// Overview:	CMICmnResources implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third party headers
#include "assert.h"

// In-house headers:
#include "MICmnResources.h"

// Instantiations:
// *** Be sure to update resource string definitions array ****
const CMICmnResources::SRsrcTextData	CMICmnResources::ms_pResourceId2TextData[] = 
{
	{ IDS_PROJNAME,								"LLDB Machine Interface Driver (MI) All rights reserved" },
	{ IDS_MI_VERSION_DESCRIPTION_DEBUG,			"Version: 1.0.0.3 (Debug)" }, // See version history in MIDriverMain.cpp
	{ IDS_MI_VERSION_DESCRIPTION,				"Version: 1.0.0.3" },
	{ IDS_MI_APPNAME_SHORT,						"MI" },
	{ IDS_MI_APPNAME_LONG,						"Machine Interface Driver" },
	{ IDS_MI_APP_FILEPATHNAME,					"Application: %s" },
	{ IDS_MI_APP_ARGS,							"Command line args: " },
	{ IDE_MI_VERSION_GDB,						"Version: GNU gdb (GDB) 7.4 \n(This is a MI stub on top of LLDB and not GDB)\nAll rights reserved.\n" }, // *** Eclipse needs this exactly!!
	{ IDS_UTIL_FILE_ERR_INVALID_PATHNAME,		"File Handler. Invalid file name path" },
	{ IDS_UTIL_FILE_ERR_OPENING_FILE,			"File Handler. Error %s opening '%s'" },
	{ IDS_UTIL_FILE_ERR_OPENING_FILE_UNKNOWN,	"File Handler. Unknown error opening '%s'" },
	{ IDE_UTIL_FILE_ERR_WRITING_FILE,			"File Handler. Error %s writing '%s'" },
	{ IDE_UTIL_FILE_ERR_WRITING_NOTOPEN,		"File Handler. File '%s' not open for write" },
	{ IDS_RESOURCES_ERR_STRING_NOT_FOUND,		"Resources. String (%d) not found in resources" },
	{ IDS_RESOURCES_ERR_STRING_TABLE_INVALID,	"Resources. String resource table is not set up" },
	{ IDS_MI_CLIENT_MSG,						"Client message: \"%s\"" },
	{ IDS_LOG_MSG_CREATION_DATE,				"Creation date %s time %s%s" },
	{ IDS_LOG_MSG_FILE_LOGGER_PATH,				"File logger path: %s%s" },	
	{ IDS_LOG_MSG_VERSION,						"Version: %s%s" },
	{ IDS_LOG_ERR_FILE_LOGGER_DISABLED,			"Log. File logger temporarily disabled due to file error '%s'" },
	{ IDS_LOG_MEDIUM_ERR_INIT,					"Log. Medium '%s' initialize failed. %s" },
	{ IDS_LOG_MEDIUM_ERR_WRITE_ANY,				"Log. Failed to write log data to any medium." },
	{ IDS_LOG_MEDIUM_ERR_WRITE_MEDIUMFAIL,		"Log. One or mediums failed writing log data." },
	{ IDS_MEDIUMFILE_NAME,						"File" },
	{ IDS_MEDIUMFILE_ERR_INVALID_PATH,			"<Invalid - not set>" },
	{ IDS_MEDIUMFILE_ERR_FILE_HEADER,			"<Invalid - header not set>" },
	{ IDS_MEDIUMFILE_NAME_LOG,					"File medium. %s" },
	{ IDE_MEDIUMFILE_ERR_GET_FILE_PATHNAME_SYS,	"File Medium. Failed to retrieve the system/executable path for the Log file" },
	{ IDE_OS_ERR_UNKNOWN,						"Unknown OS error" },
	{ IDE_OS_ERR_RETRIEVING,					"Unabled to retrieve OS error message" },
	{ IDS_DRIVERMGR_DRIVER_ERR_INIT,			"Driver Manager. Driver '%s' (ID:'%s') initialize failed. %s" },
	{ IDE_MEDIUMSTDERR_NAME,					"Stderr" },
	{ IDE_MEDIUMSTDOUT_NAME,					"Stdout" },
	{ IDE_MI_APP_EXIT_OK,						"Program exited OK" },
	{ IDE_MI_APP_EXIT_WITH_PROBLEM,				"Program exited with a problem, see %s file" },	
	{ IDE_MI_APP_ARG_USAGE,						"MI driver usage:" },
	{ IDE_MI_APP_ARG_HELP,						"-h\n--help\n\tPrints out usage information for the MI debugger." },
	{ IDE_MI_APP_ARG_VERSION,					"--version\n\tPrints out GNU (gdb) version information." },
	{ IDE_MI_APP_ARG_VERSION_LONG,				"--versionLong\n\tPrints out MI version information." },
	{ IDE_MI_APP_ARG_INTERPRETER,				"--interpreter\n\tUse the MI driver for the debugger (Default is the LLDB driver).\n\tAny LLDB command line options are ignored even if the MI driver\n\tfalls through to the LLDB driver. (Depends on MICmnConfig.h)" },
	{ IDS_STDIN_ERR_INVALID_PROMPT,				"Stdin. Invalid prompt description '%s'" },
	{ IDS_STDIN_ERR_THREAD_CREATION_FAILED,		"Stdin. Thread creation failed '%s'" },
	{ IDS_STDIN_ERR_THREAD_DELETE,				"Stdin. Thread failed to delete '%s'" },
	{ IDS_STDIN_ERR_CHKING_BYTE_AVAILABLE,		"Stdin. Peeking on stdin stream '%s'" },
	{ IDS_CMD_QUIT_HELP,						"Cmd: quit\n\tExit the MI driver application." },
	{ IDS_THREADMGR_ERR_THREAD_ID_INVALID,		"Thread Mgr. Thread ID '%s' is not valid" },
	{ IDS_THREADMGR_ERR_THREAD_FAIL_CREATE,		"Thread Mgr: Failed to create thread '%s'" },
	{ IDS_THREADMGR_ERR_THREAD_ID_NOT_FOUND,	"Thread Mgr: Thread with ID '%s' not found" },
	{ IDS_THREADMGR_ERR_THREAD_STILL_ALIVE,     "Thread Mgr: The thread(s) are still alive at Thread Mgr shutdown: %s" },
	{ IDS_FALLTHRU_DRIVER_CMD_RECEIVED,			"Fall Thru Driver. Received command '%s'. Is was %shandled" },	
	{ IDS_CMDFACTORY_ERR_INVALID_CMD_NAME,		"Command factory. MI command name '%s' is invalid" },	
	{ IDS_CMDFACTORY_ERR_INVALID_CMD_CR8FN,		"Command factory. Command creation function invalid for command '%s'. Does function exist? Pointer assigned to it?" },	
	{ IDS_CMDFACTORY_ERR_CMD_NOT_REGISTERED,	"Command factory. Command '%s' not registered" },
	{ IDS_CMDFACTORY_ERR_CMD_ALREADY_REGED,		"Command factory. Command '%s' by that name already registered" },
	{ IDS_CMDMGR_ERR_CMD_FAILED_CREATE,			"Command manager. Command creation failed. %s" },
	{ IDS_CMDMGR_ERR_CMD_INVOKER,				"Command manager. %s " },
	{ IDS_PROCESS_SIGNAL_RECEIVED,				"Process signal. Application received signal '%s' (%d)" },
	{ IDS_MI_INIT_ERR_LOG,						"Log. Error occurred during initialization %s" },
	{ IDS_MI_INIT_ERR_RESOURCES,				"Resources. Error occurred during initialization %s" },
	{ IDS_MI_INIT_ERR_INIT,						"Driver. Error occurred during initialization %s" },
	{ IDS_MI_INIT_ERR_STREAMSTDIN,				"Stdin. Error occurred during initialization %s" },
	{ IDS_MI_INIT_ERR_STREAMSTDOUT,				"Stdout. Error occurred during initialization %s" },
	{ IDS_MI_INIT_ERR_STREAMSTDERR,				"Stderr. Error occurred during initialization %s" },
	{ IDS_MI_INIT_ERR_FALLTHRUDRIVER,			"Fall Through Driver. Error occurred during initialization %s" },		
	{ IDS_MI_INIT_ERR_THREADMGR,				"Thread Mgr. Error occurred during initialization %s" },		
	{ IDS_MI_INIT_ERR_CMDINTERPRETER,			"Command interpreter. %s" },
	{ IDS_MI_INIT_ERR_CMDMGR,					"Command manager. %s" },
	{ IDS_MI_INIT_ERR_CMDFACTORY,				"Command factory. %s" },
	{ IDS_MI_INIT_ERR_CMDINVOKER,				"Command invoker. %s" },
	{ IDS_MI_INIT_ERR_CMDMONITOR,				"Command monitor. %s" },
	{ IDS_MI_INIT_ERR_LLDBDEBUGGER,				"LLDB Debugger. %s" },
	{ IDS_MI_INIT_ERR_DRIVERMGR,				"Driver manager. %s" },
	{ IDS_MI_INIT_ERR_DRIVER,					"Driver. %s" },
	{ IDS_MI_INIT_ERR_OUTOFBANDHANDLER,			"Out-of-band handler. %s " },		
	{ IDS_MI_INIT_ERR_DEBUGSESSIONINFO,			"LLDB debug session info. %s " },
	{ IDS_MI_INIT_ERR_THREADMANAGER,			"Unable to init thread manager." },
	{ IDS_CODE_ERR_INVALID_PARAMETER_VALUE,		"Code. Invalid parameter passed to function '%s'" },
	{ IDS_CODE_ERR_INVALID_PARAM_NULL_POINTER,	"Code. NULL pointer passes as a parameter to function '%s'" },
	{ IDS_LLDBDEBUGGER_ERR_INVALIDLISTENER,		"LLDB Debugger. LLDB Listener is not valid", },
	{ IDS_LLDBDEBUGGER_ERR_INVALIDDEBUGGER,		"LLDB Debugger. LLDB Debugger is not valid", },
	{ IDS_LLDBDEBUGGER_ERR_CLIENTDRIVER,		"LLDB Debugger. CMIDriverBase derived driver needs to be set prior to CMICmnLLDBDDebugger initialization" },
	{ IDS_LLDBDEBUGGER_ERR_STARTLISTENER,		"LLDB Debugger. Starting listening events for '%s' failed" },
	{ IDS_LLDBDEBUGGER_ERR_THREADCREATIONFAIL,	"LLDB Debugger. Thread creation failed '%s'" },
	{ IDS_LLDBDEBUGGER_ERR_THREAD_DELETE,		"LLDB Debugger. Thread failed to delete '%s'" },
	{ IDS_LLDBDEBUGGER_ERR_INVALIDBROADCASTER,	"LLDB Debugger. Invalid SB broadcaster class name '%s' " },
	{ IDS_LLDBDEBUGGER_ERR_INVALIDCLIENTNAME,	"LLDB Debugger. Invalid client name '%s' " },
	{ IDS_LLDBDEBUGGER_ERR_CLIENTNOTREGISTERD,	"LLDB Debugger. Client name '%s' not registered for listening events" },
	{ IDS_LLDBDEBUGGER_ERR_STOPLISTENER,		"LLDB Debugger. Failure occurred stopping event for client '%s' SBBroadcaster '%s'" },
	{ IDS_LLDBDEBUGGER_ERR_BROARDCASTER_NAME,	"LLDB Debugger. Broardcaster's name '%s' is not valid" },
	{ IDS_LLDBDEBUGGER_WRN_UNKNOWN_EVENT,		"LLDB Debugger. Unhandled event '%s'" },
	{ IDS_LLDBOUTOFBAND_ERR_UNKNOWN_EVENT,		"LLDB Out-of-band. Handling event for '%s', an event enumeration '%d' not recognized" },
	{ IDS_LLDBOUTOFBAND_ERR_PROCESS_INVALID,	"LLDB Out-of-band. Invalid '%s' in '%s'" },
	{ IDS_DBGSESSION_ERR_SHARED_DATA_RELEASE,	"LLDB debug session info. Release some or all of the data shared across command instances failed" },
	{ IDS_DBGSESSION_ERR_SHARED_DATA_ADD,		"LLDB debug session info. Failed to add '%s' data to the shared data command container" },
	{ IDS_MI_SHTDWN_ERR_LOG,					"Log. Error occurred during shutdown. %s" },
	{ IDS_MI_SHUTDOWN_ERR,						"Server shutdown failure. %s" },
	{ IDE_MI_SHTDWN_ERR_RESOURCES,				"Resources. Error occurred during shutdown. %s" },
	{ IDE_MI_SHTDWN_ERR_STREAMSTDIN,			"Stdin. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_STREAMSTDOUT,			"Stdout. Error occurred during shutdown. %s" },	
	{ IDS_MI_SHTDWN_ERR_STREAMSTDERR,			"Stderr. Error occurred during shutdown. %s" },	
	{ IDS_MI_SHTDWN_ERR_THREADMGR,				"Thread Mgr. Error occurred during shutdown. %s" },			
	{ IDS_MI_SHTDWN_ERR_CMDINTERPRETER,			"Command interpreter. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_CMDMGR,					"Command manager. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_CMDFACTORY,				"Command factory. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_CMDMONITOR,				"Command invoker. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_CMDINVOKER,				"Command monitor. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_LLDBDEBUGGER,			"LLDB Debugger. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_DRIVERMGR,				"Driver manager. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_DRIVER,					"Driver. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_OUTOFBANDHANDLER,		"Out-of-band handler. Error occurred during shutdown. %s" },
	{ IDS_MI_SHTDWN_ERR_DEBUGSESSIONINFO,		"LLDB debug session info. Error occurred during shutdown. %s" },
	{ IDE_MI_SHTDWN_ERR_THREADMANAGER,			"Unable to shutdown thread manager" },
	{ IDS_DRIVER_ERR_PARSE_ARGS,				"Driver. Driver '%s'. Parse args error '%s'" },
	{ IDS_DRIVER_ERR_PARSE_ARGS_UNKNOWN,		"Driver. Driver '%s'. Parse args error unknown" },	
	{ IDS_DRIVER_ERR_CURRENT_NOT_SET,			"Driver. Current working driver has not been set. Call CMIDriverMgr::SetUseThisDriverToDoWork()" },
	{ IDS_DRIVER_ERR_NON_REGISTERED,			"Driver. No suitable drivers registered with the CMIDriverMgr to do work" },	
	{ IDS_DRIVER_SAY_DRIVER_USING,				"Driver. Using driver '%s' internally" },	
	{ IDS_DRIVER_ERR_ID_INVALID,				"Driver. Driver '%s' invalid ID '%s'" },
	{ IDS_DRIVER_ERR_FALLTHRU_DRIVER_ERR,		"Driver. Fall through driver '%s' (ID:'%s') error '%s'" },	
	{ IDS_DRIVER_CMD_RECEIVED,					"Driver. Received command '%s'. It was %shandled%s" },	
	{ IDS_DRIVER_CMD_NOT_IN_FACTORY,			". Command '%s' not in Command Factory" },	
	{ IDS_DRIVER_WAITING_STDIN_DATA,			"Driver. Main thread suspended waiting on Stdin Monitor to resume main thread" },
	{ IDS_STDOUT_ERR_NOT_ALL_DATA_WRITTEN,		"Stdout. Not all data was written to stream. The data '%s'" },
	{ IDS_STDERR_ERR_NOT_ALL_DATA_WRITTEN,		"Stderr. Not all data was written to stream. The data '%s'" },
	{ IDS_CMD_ARGS_ERR_N_OPTIONS_REQUIRED,		"Command Args. Missing options, %d or more required" },
	{ IDS_CMD_ARGS_ERR_OPTION_NOT_FOUND,		"Command Args. Option '%s' not found" },
	{ IDS_CMD_ARGS_ERR_VALIDATION_MANDATORY,	"Command Args. Validation failed. Mandatory args not found: %s" },
	{ IDS_CMD_ARGS_ERR_VALIDATION_INVALID,		"Command Args. Validation failed. Invalid args: %s" },
	{ IDS_CMD_ARGS_ERR_VALIDATION_MAN_INVALID,  "Command Args. Validation failed. Mandatory args not found: %s. Invalid args: %s" },
	{ IDS_CMD_ARGS_ERR_VALIDATION_MISSING_INF,  "Command Args. Validation failed. Args missing additional information: %s." },
	{ IDS_CMD_ARGS_ERR_CONTEXT_NOT_ALL_EATTEN,	"Command Args. Validation failed. Not all arguments or options were recognized: %s" },
	{ IDS_WORD_INVALIDBRKTS,					"<Invalid>" },
	{ IDS_WORD_NONE,							"None" },
	{ IDS_WORD_NOT,								"not" },
	{ IDS_WORD_INVALIDEMPTY,					"<Empty>" },
	{ IDS_WORD_INVALIDNULLPTR,					"<NULL ptr>" },
	{ IDS_WORD_UNKNOWNBRKTS,					"<unkown>" },
	{ IDS_WORD_NOT_IMPLEMENTED,					"Not implemented" },	
	{ IDS_WORD_NOT_IMPLEMENTED_BRKTS,			"<not implemented>" },
	{ IDS_WORD_UNKNOWNTYPE_BRKTS,				"<unknowntype>" },
	{ IDS_CMD_ERR_N_OPTIONS_REQUIRED,			"Command '%s'. Missing options, %d required" },
	{ IDS_CMD_ERR_OPTION_NOT_FOUND,				"Command '%s'. Option '%s' not found" },
	{ IDS_CMD_ERR_ARGS,							"Command '%s'. %s" },
	{ IDS_CMD_WRN_ARGS_NOT_HANDLED,				"Command '%s'. Warning the following options not handled by the command: %s" },
	{ IDS_CMD_ERR_FNFAILED,						"Command '%s'. Fn '%s' failed" },
	{ IDS_CMD_ERR_SHARED_DATA_NOT_FOUND,		"Command '%s'. Shared data '%s' not found" },
	{ IDS_CMD_ERR_LLDBPROCESS_DETACH,			"Command '%s'. Process detach failed. '%s'"	},
	{ IDS_CMD_ERR_SETWKDIR,						"Command '%s'. Failed to set working directory '%s'" },
	{ IDS_CMD_ERR_INVALID_TARGET,				"Command '%s'. Target binary '%s' is invalid. %s" },	
	{ IDS_CMD_ERR_INVALID_TARGET_CURRENT,		"Command '%s'. Current SBTarget is invalid" },
	{ IDS_CMD_ERR_INVALID_TARGET_TYPE,			"Command '%s'. Target type '%s' is not recognized" },
	{ IDS_CMD_ERR_INVALID_TARGET_PLUGIN,		"Command '%s'. Target plugin is invalid. %s" },
	{ IDS_CMD_ERR_CONNECT_TO_TARGET,			"Command '%s'. Error connecting to target: '%s'" },
	{ IDS_CMD_ERR_INVALID_TARGETPLUGINCURRENT,	"Command '%s'. Current target plugin is invalid" },
	{ IDS_CMD_ERR_NOT_IMPLEMENTED, 				"Command '%s'. Command not implemented" },
	{ IDS_CMD_ERR_CREATE_TARGET,				"Command '%s'. Create target failed: %s" },
	{ IDS_CMD_ERR_BRKPT_LOCATION_FORMAT,		"Command '%s'. Incorrect format for breakpoint location '%s'" },
	{ IDS_CMD_ERR_BRKPT_INVALID,				"Command '%s'. Breakpoint '%s' invalid" },
	{ IDS_CMD_ERR_BRKPT_CNT_EXCEEDED,			"Command '%s'. Number of valid breakpoint exceeded %d. Cannot create new breakpoint '%s'" },
	{ IDS_CMD_ERR_SOME_ERROR,					"Command '%s'. Error: %s" },
	{ IDS_CMD_ERR_THREAD_INVALID,				"Command '%s'. Thread ID invalid" },
	{ IDS_CMD_ERR_THREAD_FRAME_RANGE_INVALID,	"Command '%s'. Thread frame range invalid" },
	{ IDS_CMD_ERR_FRAME_INVALID,				"Command '%s'. Frame ID invalid" },
	{ IDS_CMD_ERR_VARIABLE_DOESNOTEXIST,		"Command '%s'. Variable '%s' does not exist" },
	{ IDS_CMD_ERR_VARIABLE_ENUM_INVALID,		"Command '%s'. Invalid enumeration for variable '%s' formatted string '%s'" },
	{ IDS_CMD_ERR_VARIABLE_EXPRESSIONPATH,		"Command '%s'. Failed to get expression for variable '%s'" },
	{ IDS_CMD_ERR_VARIABLE_CREATION_FAILED,		"Command '%s'. Failed to create variable object for '%s'" }
};

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnResources constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnResources::CMICmnResources( void )
:	m_nResourceId2TextDataSize( 0 )
{
	// Do not use this constructor, use Initialize()
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmnResources destructor.
// Type:	Overidden.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmnResources::~CMICmnResources( void )
{
	// Do not use this destructor, use Shutdown()
}

//++ ------------------------------------------------------------------------------------
// Details:	Initialize the resources and set locality for the server.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnResources::Initialize( void )
{
	m_clientUsageRefCnt++;

	if( m_bInitialized )
		return MIstatus::success;

	m_bInitialized = ReadResourceStringData();

	return m_bInitialized;
}

//++ ------------------------------------------------------------------------------------
// Details:	Release resources for *this object.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnResources::Shutdown( void )
{
	if( --m_clientUsageRefCnt > 0 )
		return MIstatus::success;
	
	if( !m_bInitialized )
		return MIstatus::success;

	// Tear down resource explicitly
	m_mapRscrIdToTextData.clear();

	m_bInitialized = false;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Initialize the resources and set locality for the server.
// Type:	Method.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnResources::ReadResourceStringData( void )
{
	m_nResourceId2TextDataSize = sizeof ms_pResourceId2TextData / sizeof ms_pResourceId2TextData[ 0 ];
	for( MIuint i = 0; i < m_nResourceId2TextDataSize; i++ )
	{
		const SRsrcTextData * pRscrData = &ms_pResourceId2TextData[ i ];
		MapPairRscrIdToTextData_t pr( pRscrData->id, pRscrData->pTextData );
		m_mapRscrIdToTextData.insert( pr );
	}
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the corresponding text assigned to the resource ID.
// Type:	Method.
// Args:	vResourceId	- (R) MI resource ID.
// Return:	CMIUtilString - Resource text.
// Throws:	None.
//--
CMIUtilString CMICmnResources::GetString( const MIuint vResourceId ) const
{
	CMIUtilString str;
	const bool bFound = GetStringFromResource( vResourceId, str );
	assert( bFound );

	return str;
}

//++ ------------------------------------------------------------------------------------
// Details:	Determine the MI resource ID existings.
// Type:	Method.
// Args:	vResourceId	- (R) MI resource ID.
// Return:	True - Exists.
//			False - Not found.
// Throws:	None.
//--
bool CMICmnResources::HasString( const MIuint vResourceId ) const
{
	CMIUtilString str;
	return GetStringFromResource( vResourceId, str );
}

//++ ------------------------------------------------------------------------------------
// Details:	Retrieve the resource text data for the given resource ID. If a resourse ID
//			cannot be found and error is given returning the ID of the resource that 
//			cannot be located.
// Type:	Method.
// Args:	vResourceId			- (R) MI resource ID.
//			vrwResourceString	- (W) Text.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmnResources::GetStringFromResource( const MIuint vResourceId, CMIUtilString & vrwResourceString ) const
{
	MapRscrIdToTextData_t::const_iterator it = m_mapRscrIdToTextData.find( vResourceId );
	if( it == m_mapRscrIdToTextData.end() )
	{
		// Check this is a static variable init that needs this before we are ready 
		if( !m_bInitialized )
		{	
			(const_cast< CMICmnResources * >( this ))->Initialize();
			it = m_mapRscrIdToTextData.find( vResourceId );
			if( it == m_mapRscrIdToTextData.end() )
			{
				vrwResourceString = MIRSRC( IDS_RESOURCES_ERR_STRING_TABLE_INVALID );
				return MIstatus::failure;
			}
		}

		if( it == m_mapRscrIdToTextData.end() )
		{	
			vrwResourceString = CMIUtilString::Format( MIRSRC( IDS_RESOURCES_ERR_STRING_NOT_FOUND ), vResourceId );
			return MIstatus::failure;
		}
	}

	const MIuint nRsrcId( (*it).first );
	const MIchar * pRsrcData( (*it).second );

	// Return result
	vrwResourceString = pRsrcData;

	return MIstatus::success;
}
	
