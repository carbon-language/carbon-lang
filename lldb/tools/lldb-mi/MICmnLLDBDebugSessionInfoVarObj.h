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
#include <map>
#include <lldb/API/SBValue.h>

// In-house headers:
#include "MIUtilString.h"

//++ ============================================================================
// Details:	MI debug session variable object. The static functionality in *this
//			class manages a map container of *these variable objects.
// Gotchas:	None.
// Authors:	Illya Rudkin 24/03/2014.
// Changes:	None.
//--
class CMICmnLLDBDebugSessionInfoVarObj
{
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
	/* ctor */	CMICmnLLDBDebugSessionInfoVarObj( const CMIUtilString & vrStrNameReal, const CMIUtilString & vrStrName, const lldb::SBValue & vrValue, const CMIUtilString & vrStrVarObjParentName );
	/* ctor */	CMICmnLLDBDebugSessionInfoVarObj( const CMICmnLLDBDebugSessionInfoVarObj & vrOther );
	/* ctor */	CMICmnLLDBDebugSessionInfoVarObj( CMICmnLLDBDebugSessionInfoVarObj & vrOther );
	/* ctor */	CMICmnLLDBDebugSessionInfoVarObj( CMICmnLLDBDebugSessionInfoVarObj && vrOther );
	//
	CMICmnLLDBDebugSessionInfoVarObj & operator= ( const CMICmnLLDBDebugSessionInfoVarObj & vrOther );
	CMICmnLLDBDebugSessionInfoVarObj & operator= ( CMICmnLLDBDebugSessionInfoVarObj && vrwOther );
	//
	const CMIUtilString &	GetName( void ) const;
	const CMIUtilString &	GetNameReal( void ) const;
	const CMIUtilString &	GetValueFormatted( void ) const;
	const lldb::SBValue &	GetValue( void ) const;
	varType_e				GetType( void ) const;
	bool					SetVarFormat( const varFormat_e veVarFormat );
	const CMIUtilString &	GetVarParentName( void ) const;
	void					UpdateValue( void );

// Overridden:
public:
	// From CMICmnBase
	/* dtor */ virtual ~CMICmnLLDBDebugSessionInfoVarObj( void );

// Typedefs:
private:
	typedef std::map< CMIUtilString, CMICmnLLDBDebugSessionInfoVarObj >		MapKeyToVarObj_t;			
	typedef std::pair< CMIUtilString, CMICmnLLDBDebugSessionInfoVarObj >	MapPairKeyToVarObj_t;

// Statics:
private:
	static CMIUtilString 	GetStringFormatted( const MIuint64 vnValue, const MIchar * vpStrValueNatural, varFormat_e veVarFormat );
	
// Methods:
private:
	bool	CopyOther( const CMICmnLLDBDebugSessionInfoVarObj & vrOther );
	bool	MoveOther( CMICmnLLDBDebugSessionInfoVarObj & vrwOther );

// Attributes:
private:
	static const MIchar *	ms_aVarFormatStrings[];
	static const MIchar *	ms_aVarFormatChars[];
	static MapKeyToVarObj_t	ms_mapVarIdToVarObj;
	static MIuint			ms_nVarUniqueId;
	//
	// *** Upate the copy move constructors and assignment operator ***
	varFormat_e		m_eVarFormat;
	varType_e		m_eVarType;
	CMIUtilString	m_strName;
	lldb::SBValue	m_SBValue;
	CMIUtilString	m_strNameReal;
	CMIUtilString	m_strFormattedValue;
	CMIUtilString	m_strVarObjParentName;
	// *** Upate the copy move constructors and assignment operator ***
};
