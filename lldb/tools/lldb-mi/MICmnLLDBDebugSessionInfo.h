//===-- MICmnLLDBDebugSessionInfo.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmnLLDBDebugSessionInfo.h
//
// Overview:	CMICmnLLDBDebugSessionInfo interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Third party headers:
#include <map>
#include <vector>
#include <lldb/API/SBDebugger.h>
#include <lldb/API/SBListener.h> 
#include <lldb/API/SBProcess.h>
#include <lldb/API/SBTarget.h>

// In-house headers:
#include "MICmnBase.h"
#include "MIUtilSingletonBase.h"
#include "MICmnLLDBDebugSessionInfoVarObj.h"
#include "MICmnMIValueTuple.h"
#include "MIUtilMapIdToVariant.h"

// Declarations:
class CMICmnLLDBDebugger;
struct SMICmdData;
class CMICmnMIValueTuple;
class CMICmnMIValueList;

//++ ============================================================================
// Details:	MI debug session object that holds debugging information between
//			instances of MI commands executing their work and producing MI
//			result records. Information/data is set by one or many commands then
//			retrieved by the same or other sebsequent commands.
//			It primarily to hold LLDB type objects.
//			A singleton class.
// Gotchas:	None.
// Authors:	Illya Rudkin 04/03/2014.
// Changes:	None.
//--
class CMICmnLLDBDebugSessionInfo
:	public CMICmnBase
,	public MI::ISingleton< CMICmnLLDBDebugSessionInfo >
{
	friend class MI::ISingleton< CMICmnLLDBDebugSessionInfo >;

// Structs:
public:
	//++ ============================================================================
	// Details:	Break point information object. Used to easily pass information about
	//			a break around and record break point information to be recalled by
	//			other commands or LLDB event handling functions.
	//-- 
	struct SBrkPtInfo
	{
		SBrkPtInfo( void )
		:	m_id( 0 )					
		,	m_bDisp( false )					
		,	m_bEnabled( false )				
		,	m_pc( 0 )						
		,	m_nLine( 0 )					
		,	m_bHaveArgOptionThreadGrp( false )	
		,	m_nTimes( 0 )					
		,	m_bPending( false )					
		,	m_nIgnore( 0 )		
		,	m_bCondition( false )
		,	m_bBrkPtThreadId( false )
		,	m_nBrkPtThreadId( 0 )
		{
		}

		MIuint			m_id;						// LLDB break point ID.
		CMIUtilString	m_strType;					// Break point type. 
		bool			m_bDisp	;					// True = "del", false = "keep".
		bool			m_bEnabled;					// True = enabled, false = disabled break point.
		MIuint			m_pc;						// Address number.
		CMIUtilString	m_fnName;					// Function name.
		CMIUtilString	m_fileName;					// File name text.
		CMIUtilString	m_path;						// Full file name and path text.
		MIuint			m_nLine;					// File line number.
		bool			m_bHaveArgOptionThreadGrp;	// True = include MI field, false = do not include "thread-groups".
		CMIUtilString	m_strOptThrdGrp;			// Thread group number.
		MIuint			m_nTimes;					// The count of the breakpoint existence.
		CMIUtilString	m_strOrigLoc;				// The name of the break point.
		bool			m_bPending;					// True = the breakpoint has not been established yet, false = location found
		MIuint			m_nIgnore;					// The number of time the breakpoint is run over before it is stopped on a hit 
		bool			m_bCondition;				// True = break point is conditional, use condition expression, false = no condition
		CMIUtilString	m_strCondition;				// Break point condition expression
		bool			m_bBrkPtThreadId;			// True = break point is specified to work with a specific thread, false = no specified thread given
		MIuint			m_nBrkPtThreadId;			// Restrict the breakpoint to the specified thread-id
	};

// Typedefs:
public:
	typedef std::vector< uint32_t >	VecActiveThreadId_t;

// Methods:
public:
	bool	Initialize( void );
	bool	Shutdown( void );

	// Variant type data which can be assigned and retrieved across all command instances
	template< typename T > 
	bool	SharedDataAdd( const CMIUtilString & vKey, const T & vData );
	template< typename T > 
	bool	SharedDataRetrieve( const CMIUtilString & vKey, T & vwData );
	bool	SharedDataDestroy( void );
	
	//	Common command required functionality
	bool	AccessPath( const CMIUtilString & vPath, bool & vwbYesAccessible );
	bool	GetFrameInfo( const lldb::SBFrame & vrFrame, lldb::addr_t & vwPc, CMIUtilString & vwFnName, CMIUtilString & vwFileName, CMIUtilString & vwPath, MIuint & vwnLine );
	bool	GetThreadFrames( const SMICmdData & vCmdData, const MIuint vThreadIdx, CMIUtilString & vwrThreadFrames );
	bool	GetThreadFrames2( const SMICmdData & vCmdData, const MIuint vThreadIdx, CMIUtilString & vwrThreadFrames );
	bool	ResolvePath( const SMICmdData & vCmdData, const CMIUtilString & vPath, CMIUtilString & vwrResolvedPath );
	bool	ResolvePath( const CMIUtilString & vstrUnknown, CMIUtilString & vwrResolvedPath );
	bool	MIResponseFormFrameInfo( const lldb::SBThread & vrThread, const MIuint vnLevel, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	MIResponseFormFrameInfo( const lldb::addr_t vPc, const CMIUtilString & vFnName, const CMIUtilString & vFileName, const CMIUtilString & vPath, const MIuint vnLine, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	MIResponseFormFrameInfo2( const lldb::addr_t vPc, const CMIUtilString & vArgInfo, const CMIUtilString & vFnName, const CMIUtilString & vFileName, const CMIUtilString & vPath, const MIuint vnLine, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	MIResponseFormThreadInfo( const SMICmdData & vCmdData, const lldb::SBThread & vrThread, CMICmnMIValueTuple & vwrMIValueTuple );
	bool	MIResponseFormThreadInfo2( const SMICmdData & vCmdData, const lldb::SBThread & vrThread, CMICmnMIValueTuple & vwrMIValueTuple );
	bool	MIResponseFormThreadInfo3( const SMICmdData & vCmdData, const lldb::SBThread & vrThread, CMICmnMIValueTuple & vwrMIValueTuple );
	bool	MIResponseFormVariableInfo( const lldb::SBFrame & vrFrame, const MIuint vMaskVarTypes, CMICmnMIValueList & vwrMiValueList );
	bool	MIResponseFormVariableInfo2( const lldb::SBFrame & vrFrame, const MIuint vMaskVarTypes, CMICmnMIValueList & vwrMiValueList );
	bool	MIResponseFormVariableInfo3( const lldb::SBFrame & vrFrame, const MIuint vMaskVarTypes, CMICmnMIValueList & vwrMiValueList );
	bool	MIResponseFormBrkPtFrameInfo( const SBrkPtInfo & vrBrkPtInfo, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	MIResponseFormBrkPtInfo( const SBrkPtInfo & vrBrkPtInfo, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	GetBrkPtInfo( const lldb::SBBreakpoint & vBrkPt, SBrkPtInfo & vrwBrkPtInfo ) const;
	bool	RecordBrkPtInfo( const MIuint vnBrkPtId, const SBrkPtInfo & vrBrkPtInfo );
	bool	RecordBrkPtInfoGet( const MIuint vnBrkPtId, SBrkPtInfo & vrwBrkPtInfo ) const;
	bool	RecordBrkPtInfoDelete( const MIuint vnBrkPtId );

// Attributes:
public:
	// The following are available to all command instances
	lldb::SBDebugger &		m_rLldbDebugger;			
	lldb::SBListener &		m_rLlldbListener;	
	lldb::SBTarget 			m_lldbTarget;
    lldb::SBProcess			m_lldbProcess;
	const MIuint			m_nBrkPointCntMax;
  	VecActiveThreadId_t		m_vecActiveThreadId;
	lldb::tid_t				m_currentSelectedThread;

	// These are keys that can be used to access the shared data map
	// Note: This list is expected to grow and will be moved and abstracted in the future.
	const CMIUtilString	m_constStrSharedDataKeyWkDir;
    const CMIUtilString m_constStrSharedDataSolibPath;

// Typedefs:
private:
	typedef std::vector< CMICmnLLDBDebugSessionInfoVarObj >	VecVarObj_t;
	typedef std::map< MIuint, SBrkPtInfo >					MapBrkPtIdToBrkPtInfo_t;		
	typedef std::pair< MIuint, SBrkPtInfo >					MapPairBrkPtIdToBrkPtInfo_t;

// Methods:
private:
	/* ctor */	CMICmnLLDBDebugSessionInfo( void );
	/* ctor */	CMICmnLLDBDebugSessionInfo( const CMICmnLLDBDebugSessionInfo & );
	void		operator=( const CMICmnLLDBDebugSessionInfo & );
	//
	bool	GetVariableInfo( const MIuint vnMaxDepth, const lldb::SBValue & vrValue, const bool vbIsChildValue, CMICmnMIValueList & vwrMiValueList, MIuint & vrwnDepth );
	bool	GetVariableInfo2( const MIuint vnMaxDepth, const lldb::SBValue & vrValue, const bool vbIsChildValue, CMICmnMIValueList & vwrMiValueList, MIuint & vrwnDepth );
	
// Overridden:
private:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmnLLDBDebugSessionInfo( void );

// Attributes:
private:
	CMIUtilMapIdToVariant	m_mapIdToSessionData;		// Hold and retrieve key to value data available across all commands
	VecVarObj_t				m_vecVarObj;				// Vector of session variable objects
	MapBrkPtIdToBrkPtInfo_t	m_mapBrkPtIdToBrkPtInfo;
};

//++ ------------------------------------------------------------------------------------
// Details:	Command instances can create and share data between other instances of commands.
//			This function adds new data to the shared data. Using the same ID more than
//			once replaces any previous matching data keys.
// Type:	Template method.
// Args:	T		- The type of the object to be stored.
//			vKey	- (R) A non empty unique data key to retrieve the data by.
//			vData	- (R) Data to be added to the share.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
template< typename T > 
bool CMICmnLLDBDebugSessionInfo::SharedDataAdd( const CMIUtilString & vKey, const T & vData )
{
	if( !m_mapIdToSessionData.Add< T >( vKey, vData ) )
	{
		SetErrorDescription( m_mapIdToSessionData.GetErrorDescription() );
		return MIstatus::failure;
	}

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Command instances can create and share data between other instances of commands.
//			This function retrieves data from the shared data container.
// Type:	Method.
// Args:	T		- The type of the object being retrieved.
//			vKey	- (R) A non empty unique data key to retrieve the data by.
//			vData	- (W) The data.
// Return:	bool - True = data found, false = data not found or an error occurred trying to fetch.
// Throws:	None.
//--
template< typename T > 
bool CMICmnLLDBDebugSessionInfo::SharedDataRetrieve( const CMIUtilString & vKey, T & vwData )
{
	bool bDataFound = false;

	if( !m_mapIdToSessionData.Get< T >( vKey, vwData, bDataFound ) )
	{
		SetErrorDescription( m_mapIdToSessionData.GetErrorDescription() );
		return MIstatus::failure;
	}

	return bDataFound;
}