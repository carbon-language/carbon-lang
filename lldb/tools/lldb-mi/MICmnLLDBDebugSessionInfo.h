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

// Typedefs:
public:
	typedef std::vector< uint32_t >	VecActiveThreadId_t;

// Methods:
public:
	bool	Initialize( void );
	bool	Shutdown( void );

	// Variant type data which can be assigned and retrieved across all command instances
	bool	SharedDataAdd( const CMIUtilString & vKey, const CMIUtilString & vData );
	bool	SharedDataRetrieve( const CMIUtilString & vKey, CMIUtilString & vwData );
	bool	SharedDataDestroy( void );
	
	//	Common command required functionality
	bool	AccessPath( const CMIUtilString & vPath, bool & vwbYesAccessible );
	bool	GetFrameInfo( const lldb::SBFrame & vrFrame, lldb::addr_t & vwPc, CMIUtilString & vwFnName, CMIUtilString & vwFileName, CMIUtilString & vwPath, MIuint & vwnLine );
	bool	GetThreadFrames( const SMICmdData & vCmdData, const MIuint vThreadIdx, CMICmnMIValueTuple & vwrThreadFrames );
	bool	ResolvePath( const SMICmdData & vCmdData, const CMIUtilString & vPath, CMIUtilString & vwrResolvedPath );
	bool	ResolvePath( const CMIUtilString & vstrUnknown, CMIUtilString & vwrResolvedPath );
	bool	MIResponseFormFrameInfo( const lldb::SBThread & vrThread, const MIuint vnLevel, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	MIResponseFormFrameInfo( const lldb::addr_t vPc, const CMIUtilString & vFnName, const CMIUtilString & vArgs, const CMIUtilString & vFileName, const CMIUtilString & vPath, const MIuint vnLine, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	MIResponseFormThreadInfo( const SMICmdData & vCmdData, const lldb::SBThread & vrThread, CMICmnMIValueTuple & vwrMIValueTuple );
	bool	MIResponseFormVariableInfo( const lldb::SBFrame & vrFrame, const MIuint vMaskVarTypes, CMICmnMIValueList & vwrMiValueList );
	bool	MIResponseFormBrkPtFrameInfo( const lldb::addr_t vPc, const CMIUtilString & vFnName, const CMIUtilString & vFileName, const CMIUtilString & vPath, const MIuint vnLine, CMICmnMIValueTuple & vwrMiValueTuple );
	bool	MIResponseFormBrkPtInfo( const lldb::break_id_t vId, const CMIUtilString & vStrType, const bool vbDisp, const bool vbEnabled, const lldb::addr_t vPc, const CMIUtilString & vFnName, const CMIUtilString & vFileName, const CMIUtilString & vPath, const MIuint vnLine, const bool vbHaveArgOptionThreadGrp, const CMIUtilString & vStrOptThrdGrp, const MIuint & vnTimes, const CMIUtilString & vStrOrigLoc, CMICmnMIValueTuple & vwrMiValueTuple );

	// Attributes:
public:
	// The following are available to all command instances
	lldb::SBDebugger &		m_rLldbDebugger;			
	lldb::SBListener &		m_rLlldbListener;	
	lldb::SBTarget 			m_lldbTarget;
    lldb::SBProcess			m_lldbProcess;
	MIuint					m_nBrkPointCnt;
	const MIuint			m_nBrkPointCntMax;
  	VecActiveThreadId_t		m_vecActiveThreadId;
	lldb::tid_t				m_currentSelectedThread;
	//
	const CMIUtilString	m_constStrSharedDataKeyWkDir;

// Typedefs:
private:
	typedef std::map< CMIUtilString, CMIUtilString >		MapKeyToStringValue_t;		// Todo: change this to be a variant type
	typedef std::pair< CMIUtilString, CMIUtilString >		MapPairKeyToStringValue_t;
	typedef std::vector< CMICmnLLDBDebugSessionInfoVarObj >	VecVarObj_t;

// Methods:
private:
	/* ctor */	CMICmnLLDBDebugSessionInfo( void );
	/* ctor */	CMICmnLLDBDebugSessionInfo( const CMICmnLLDBDebugSessionInfo & );
	void		operator=( const CMICmnLLDBDebugSessionInfo & );

// Overridden:
private:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmnLLDBDebugSessionInfo( void );

// Attributes:
private:
	MapKeyToStringValue_t	m_mapKeyToStringValue;		// Hold and retrieve key to value data available across all commands
	VecVarObj_t				m_vecVarObj;				// Vector of session variable objects
};
