//===-- MICmnLLDBDebugSessionInfoVarObj.h -----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmnLLDBDebugSessionInfoVarObj.h
//
// Overview:	CMICmnLLDBDebugSessionInfoVarObj interface.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

#pragma once

// Third Party Headers:
#include <vector>
#include <map>
#include <lldb/API/SBValue.h>

// In-house headers:
#include "MIUtilString.h"

//++ ============================================================================
// Details:	MI debug session variable object.
// Gotchas:	None.
// Authors:	Illya Rudkin 24/03/2014.
// Changes:	None.
//--
class CMICmnLLDBDebugSessionInfoVarObj
{
// Typedefs:
public:
	typedef std::vector< CMICmnLLDBDebugSessionInfoVarObj >	VecVarObj_t;

// Enums:
public:
	//++ ----------------------------------------------------------------------
	// Details:	Enumeration of a variable type that is not a composite type
	//--
	enum varFormat_e
	{
		// CODETAG_SESSIONINFO_VARFORMAT_ENUM
		// *** Order is import here ***
		eVarFormat_Invalid = 0,
		eVarFormat_Binary,
		eVarFormat_Octal,
		eVarFormat_Decimal,
		eVarFormat_Hex,
		eVarFormat_Natural,
		eVarFormat_count	// Always last one
	};

	//++ ----------------------------------------------------------------------
	// Details:	Enumeration of a variable type by composite or internal type
	//--
	enum varType_e
	{
		eVarType_InValid = 0,
		eVarType_Composite,		// i.e. struct
		eVarType_Internal,		// i.e. int 
		eVarType_count			// Always last one
	};

// Statics:
public:
	static varFormat_e		GetVarFormatForString( const CMIUtilString & vrStrFormat );
	static varFormat_e		GetVarFormatForChar( const MIchar & vrcFormat );
	static CMIUtilString	GetValueStringFormatted( const lldb::SBValue & vrValue, const varFormat_e veVarFormat );
	static void				VarObjAdd( const CMICmnLLDBDebugSessionInfoVarObj & vrVarObj );
	static void				VarObjDelete( const CMIUtilString & vrVarName );
	static bool				VarObjGet( const CMIUtilString & vrVarName, CMICmnLLDBDebugSessionInfoVarObj & vrwVarObj );
	static void				VarObjUpdate( const CMICmnLLDBDebugSessionInfoVarObj & vrVarObj );
	static void				VarObjIdInc( void );
	static MIuint			VarObjIdGet( void );
	static void				VarObjIdResetToZero( void );
	static void				VarObjClear( void );

// Methods:
public:
	/* ctor */	CMICmnLLDBDebugSessionInfoVarObj( void );
	/* ctor */	CMICmnLLDBDebugSessionInfoVarObj( const CMIUtilString & vrStrNameReal, const CMIUtilString & vrStrName, const lldb::SBValue & vrValue );
	//
	const CMIUtilString &	GetName( void ) const;
	const CMIUtilString &	GetNameReal( void ) const;
	const CMIUtilString &	GetValueFormatted( void ) const;
	const lldb::SBValue &	GetValue( void ) const;
	varType_e				GetType( void ) const;
	bool					SetVarFormat( const varFormat_e veVarFormat );
	void					UpdateValue( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmnLLDBDebugSessionInfoVarObj( void );

// Typedefs:
private:
	typedef std::map< CMIUtilString, CMICmnLLDBDebugSessionInfoVarObj * >	MapVarRealNameToVarObject_t;		// ToDo: Do I need this?
	typedef std::pair< CMIUtilString, CMICmnLLDBDebugSessionInfoVarObj * >	MapPairVarRealNameToVarObject_t;	// ToDo: Do I need this?

// Statics:
private:
	static bool	MapVarObjAdd( const CMIUtilString & vrVarRealName, const CMICmnLLDBDebugSessionInfoVarObj & vrVarObj );
	static bool	MapVarObjDelete( const CMIUtilString & vrVarName );

// Attributes:
private:
	static const MIchar *	ms_aVarFormatStrings[];
	static const MIchar *	ms_aVarFormatChars[];
	static VecVarObj_t		ms_vecVarObj;			// ToDo: Replace this vector container for something more efficient (set will not compile)
	static MIuint			ms_nVarUniqueId;
	//
	varFormat_e		m_eVarFormat;
	varType_e		m_eVarType;
	CMIUtilString	m_strName;
	lldb::SBValue	m_SBValue;
	CMIUtilString	m_strNameReal;
	CMIUtilString	m_strFormattedValue;
};
