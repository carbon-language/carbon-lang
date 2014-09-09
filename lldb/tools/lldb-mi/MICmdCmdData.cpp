//===-- MICmdCmdData.cpp ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//++
// File:		MICmdCmdData.cpp
//
// Overview:	CMICmdCmdDataEvaluateExpression		implementation.
//				CMICmdCmdDataDisassemble			implementation.
//				CMICmdCmdDataReadMemoryBytes		implementation.
//				CMICmdCmdDataReadMemory				implementation.
//				CMICmdCmdDataListRegisterNames		implementation.
//				CMICmdCmdDataListRegisterValues		implementation.
//				CMICmdCmdDataListRegisterChanged	implementation.
//				CMICmdCmdDataWriteMemoryBytes		implementation.
//				CMICmdCmdDataWriteMemory			implementation.
//
// Environment:	Compilers:	Visual C++ 12.
//							gcc (Ubuntu/Linaro 4.8.1-10ubuntu9) 4.8.1
//				Libraries:	See MIReadmetxt. 
//
// Copyright:	None.
//--

// Third Party Headers:
#include <lldb/API/SBThread.h>
#include <lldb/API/SBInstruction.h>
#include <lldb/API/SBInstructionList.h>
#include <lldb/API/SBStream.h>

// In-house headers:
#include "MICmdCmdData.h"
#include "MICmnMIResultRecord.h"
#include "MICmnMIValueConst.h"
#include "MICmnLLDBDebugger.h"
#include "MICmnLLDBDebugSessionInfo.h"
#include "MICmnLLDBProxySBValue.h"
#include "MICmdArgValNumber.h"
#include "MICmdArgValString.h"
#include "MICmdArgValThreadGrp.h"
#include "MICmdArgValOptionLong.h"
#include "MICmdArgValOptionShort.h"
#include "MICmdArgValListOfN.h"
#include "MICmdArgValConsume.h"
#include "MICmnLLDBDebugSessionInfoVarObj.h"
#include "MICmnLLDBUtilSBValue.h"

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataEvaluateExpression constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataEvaluateExpression::CMICmdCmdDataEvaluateExpression( void )
:	m_bExpressionValid( true )
,	m_bEvaluatedExpression( true )
,	m_strValue( "??" )
,	m_bCompositeVarType( false )
,	m_bFoundInvalidChar( false )
,	m_cExpressionInvalidChar( 0x00 )
,	m_constStrArgThread( "thread" )
,	m_constStrArgFrame( "frame" )
,	m_constStrArgExpr( "expr" )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-evaluate-expression";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataEvaluateExpression::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataEvaluateExpression destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataEvaluateExpression::~CMICmdCmdDataEvaluateExpression( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataEvaluateExpression::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThread, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgFrame, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgExpr, true, true, true, true ) ) );
	return (bOk && ParseValidateCmdOptions() );
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataEvaluateExpression::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgExpr, String, m_constStrArgExpr );

	const CMIUtilString & rExpression( pArgExpr->GetValue() );
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	lldb::SBThread thread = rProcess.GetSelectedThread();
	m_bExpressionValid = (thread.GetNumFrames() > 0);
	if( !m_bExpressionValid )
		return MIstatus::success;
	
	lldb::SBFrame frame = thread.GetSelectedFrame();
	lldb::SBValue value = frame.EvaluateExpression( rExpression.c_str() );
	if( !value.IsValid() )
		value = frame.FindVariable( rExpression.c_str() );
	if( !value.IsValid() )
	{
		m_bEvaluatedExpression = false;
		return MIstatus::success;
	}
	const CMICmnLLDBUtilSBValue utilValue( value );
	if( !utilValue.HasName() )
	{
		if( HaveInvalidCharacterInExpression( rExpression, m_cExpressionInvalidChar ) )
		{
			m_bFoundInvalidChar = true;
			return MIstatus::success;
		}

		m_strValue = rExpression;
		return MIstatus::success;
	}
	if( rExpression.IsQuoted() )
	{
		m_strValue = rExpression.Trim( '\"' );
		return MIstatus::success;
	}

	MIuint64 nNumber = 0;
	if( CMICmnLLDBProxySBValue::GetValueAsUnsigned( value, nNumber ) == MIstatus::success )
	{
		const lldb::ValueType eValueType = value.GetValueType(); MIunused( eValueType );
		m_strValue = utilValue.GetValue();
		CMIUtilString strCString;
		if( CMICmnLLDBProxySBValue::GetCString( value, strCString ) )
		{
			m_strValue += CMIUtilString::Format( " '%s'", strCString.c_str() );
		}
		return MIstatus::success;
	}

	// Composite type i.e. struct
	m_bCompositeVarType = true;
	const MIuint nChild = value.GetNumChildren();
	for( MIuint i = 0; i < nChild; i++ )
	{
		lldb::SBValue member = value.GetChildAtIndex( i );
		const bool bValid = member.IsValid();
		CMIUtilString strType( MIRSRC( IDS_WORD_UNKNOWNTYPE_BRKTS ) );
		if( bValid )
		{
			const CMIUtilString strValue( CMICmnLLDBDebugSessionInfoVarObj::GetValueStringFormatted( member, CMICmnLLDBDebugSessionInfoVarObj::eVarFormat_Natural ) );
			const char * pTypeName = member.GetName();
			if( pTypeName != nullptr )
				strType = pTypeName;

			// MI print "{variable = 1, variable2 = 3, variable3 = 5}"
			const bool bNoQuotes = true;
			const CMICmnMIValueConst miValueConst( strValue, bNoQuotes );
			const bool bUseSpaces = true;
			const CMICmnMIValueResult miValueResult( strType, miValueConst, bUseSpaces );
			m_miValueTuple.Add( miValueResult, bUseSpaces );
		}
	}
		
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataEvaluateExpression::Acknowledge( void )
{
	if( m_bExpressionValid )
	{
		if( m_bEvaluatedExpression )
		{
			if( m_bCompositeVarType )
			{
				const CMICmnMIValueConst miValueConst( m_miValueTuple.GetString() );
				const CMICmnMIValueResult miValueResult( "value", miValueConst );
				const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
				m_miResultRecord = miRecordResult;
				return MIstatus::success;
			}

			if( m_bFoundInvalidChar )
			{
				const CMICmnMIValueConst miValueConst( CMIUtilString::Format( "Invalid character '%c' in expression", m_cExpressionInvalidChar ) );
				const CMICmnMIValueResult miValueResult( "msg", miValueConst );
				const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
				m_miResultRecord = miRecordResult;
				return MIstatus::success;
			}
			
			const CMICmnMIValueConst miValueConst( m_strValue );
			const CMICmnMIValueResult miValueResult( "value", miValueConst );
			const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
			m_miResultRecord = miRecordResult;
			return MIstatus::success;
		}

		const CMICmnMIValueConst miValueConst( "Could not evaluate expression" );
		const CMICmnMIValueResult miValueResult( "msg", miValueConst );
		const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
		m_miResultRecord = miRecordResult;
		return MIstatus::success;
	}

	const CMICmnMIValueConst miValueConst( "Invalid expression" );
	const CMICmnMIValueResult miValueResult( "msg", miValueConst );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
	m_miResultRecord = miRecordResult;
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataEvaluateExpression::CreateSelf( void )
{
	return new CMICmdCmdDataEvaluateExpression();
}

//++ ------------------------------------------------------------------------------------
// Details:	Examine the expression string to see if it contains invalid characters.
// Type:	Method.
// Args:	vrExpr			- (R) Expression string given to *this command.
//			vrwInvalidChar	- (W) True = Invalid character found, false = nothing found.
// Return:	bool - True = Invalid character found, false = nothing found.
// Throws:	None.
//--
bool CMICmdCmdDataEvaluateExpression::HaveInvalidCharacterInExpression( const CMIUtilString & vrExpr, MIchar & vrwInvalidChar )
{
	bool bFoundInvalidCharInExpression = false;
	vrwInvalidChar = 0x00;

	if( vrExpr.at( 0 ) == '\\' )
	{
		// Example: Mouse hover over "%5d" expression has \"%5d\" in it 
		bFoundInvalidCharInExpression = true;
		vrwInvalidChar = '\\';
	}
		
	return bFoundInvalidCharInExpression;
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataDisassemble constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataDisassemble::CMICmdCmdDataDisassemble( void )
:	m_constStrArgThread( "thread" )
,	m_constStrArgAddrStart( "s" )	
,	m_constStrArgAddrEnd( "e" )	
,	m_constStrArgConsume( "--" )
,	m_constStrArgMode( "mode" )	
,	m_miValueList( true )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-disassemble";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataDisassemble::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataDisassemble destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataDisassemble::~CMICmdCmdDataDisassemble( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataDisassemble::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThread, true, true, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgAddrStart, true, true, CMICmdArgValListBase::eArgValType_StringQuotedNumber, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgAddrEnd, true, true, CMICmdArgValListBase::eArgValType_StringQuotedNumber, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValConsume( m_constStrArgConsume, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValNumber( m_constStrArgMode, true, true ) ) );
	return (bOk && ParseValidateCmdOptions() );
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataDisassemble::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgThread, OptionLong, m_constStrArgThread );
	CMICMDBASE_GETOPTION( pArgAddrStart, OptionShort, m_constStrArgAddrStart );
	CMICMDBASE_GETOPTION( pArgAddrEnd, OptionShort, m_constStrArgAddrEnd );
	CMICMDBASE_GETOPTION( pArgMode, Number, m_constStrArgMode );

	// Retrieve the --thread option's thread ID (only 1)
	MIuint64 nThreadId = UINT64_MAX;
	if( !pArgThread->GetExpectedOption< CMICmdArgValNumber, MIuint64 >( nThreadId ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_THREAD_INVALID ), m_cmdData.strMiCmd.c_str(), m_constStrArgThread.c_str() ) );
		return MIstatus::failure;
	}
	CMIUtilString strAddrStart;
	if( !pArgAddrStart->GetExpectedOption< CMICmdArgValString, CMIUtilString >( strAddrStart ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_DISASM_ADDR_START_INVALID ), m_cmdData.strMiCmd.c_str(), m_constStrArgAddrStart.c_str() ) );
		return MIstatus::failure;
	}
	MIint64 nAddrStart = 0;
	if( !strAddrStart.ExtractNumber( nAddrStart ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_DISASM_ADDR_START_INVALID ), m_cmdData.strMiCmd.c_str(), m_constStrArgAddrStart.c_str() ) );
		return MIstatus::failure;
	}
	
	CMIUtilString strAddrEnd;
	if( !pArgAddrEnd->GetExpectedOption< CMICmdArgValString, CMIUtilString >( strAddrEnd ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_DISASM_ADDR_END_INVALID ), m_cmdData.strMiCmd.c_str(), m_constStrArgAddrEnd.c_str() ) );
		return MIstatus::failure;
	}
	MIint64 nAddrEnd = 0;
	if( !strAddrEnd.ExtractNumber( nAddrEnd ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_DISASM_ADDR_END_INVALID ), m_cmdData.strMiCmd.c_str(), m_constStrArgAddrEnd.c_str() ) );
		return MIstatus::failure;
	}
	const MIuint nDisasmMode = pArgMode->GetValue();

	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBTarget & rTarget = rSessionInfo.m_lldbTarget;
	lldb::addr_t lldbStartAddr = static_cast< lldb::addr_t >( nAddrStart );
	lldb::SBInstructionList instructions = rTarget.ReadInstructions( lldb::SBAddress( lldbStartAddr, rTarget ), nAddrEnd - nAddrStart );
	const MIuint nInstructions = instructions.GetSize();
	for( size_t i = 0; i < nInstructions; i++ )
	{
		const MIchar * pUnknown = "??";
		lldb::SBInstruction instrt = instructions.GetInstructionAtIndex( i );
		const MIchar * pStrMnemonic = instrt.GetMnemonic( rTarget );
		pStrMnemonic = (pStrMnemonic != nullptr) ? pStrMnemonic : pUnknown;
		lldb::SBAddress address = instrt.GetAddress();
		lldb::addr_t addr = address.GetLoadAddress( rTarget );
		const MIchar * pFnName = address.GetFunction().GetName();
		pFnName = (pFnName != nullptr) ? pFnName : pUnknown;
		lldb::addr_t addrOffSet = address.GetOffset();
		const MIchar * pStrOperands = instrt.GetOperands( rTarget );
		pStrOperands = (pStrOperands != nullptr) ? pStrOperands : pUnknown;

		// MI "{address=\"0x%08llx\",func-name=\"%s\",offset=\"%lld\",inst=\"%s %s\"}"
		const CMICmnMIValueConst miValueConst( CMIUtilString::Format( "0x%08llx", addr ) );
		const CMICmnMIValueResult miValueResult( "address", miValueConst );
		CMICmnMIValueTuple miValueTuple( miValueResult );
		const CMICmnMIValueConst miValueConst2( pFnName );
		const CMICmnMIValueResult miValueResult2( "func-name", miValueConst2 );
		miValueTuple.Add( miValueResult2 );
		const CMICmnMIValueConst miValueConst3( CMIUtilString::Format( "0x%lld", addrOffSet ) );
		const CMICmnMIValueResult miValueResult3( "offset", miValueConst3 );
		miValueTuple.Add( miValueResult3 );
		const CMICmnMIValueConst miValueConst4( CMIUtilString::Format( "%s %s", pStrMnemonic, pStrOperands ) );
		const CMICmnMIValueResult miValueResult4( "inst", miValueConst4 );
		miValueTuple.Add( miValueResult4 );
		
		if( nDisasmMode == 1 )
		{
			lldb::SBLineEntry lineEntry = address.GetLineEntry();
			const MIuint nLine = lineEntry.GetLine();
			const MIchar * pFileName = lineEntry.GetFileSpec().GetFilename();
			pFileName = (pFileName != nullptr) ? pFileName : pUnknown;

			// MI "src_and_asm_line={line=\"%u\",file=\"%s\",line_asm_insn=[ ]}"
			const CMICmnMIValueConst miValueConst( CMIUtilString::Format( "0x%u", nLine ) );
			const CMICmnMIValueResult miValueResult( "line", miValueConst );
			CMICmnMIValueTuple miValueTuple2( miValueResult );
			const CMICmnMIValueConst miValueConst2( pFileName );
			const CMICmnMIValueResult miValueResult2( "file", miValueConst2 );
			miValueTuple2.Add( miValueResult2 );		
			const CMICmnMIValueList miValueList( miValueTuple );
			const CMICmnMIValueResult miValueResult3( "line_asm_insn", miValueList );
			miValueTuple2.Add( miValueResult3 );
			const CMICmnMIValueResult miValueResult4( "src_and_asm_line", miValueTuple2 );
			m_miValueList.Add( miValueResult4 );
		}
		else
		{
			m_miValueList.Add( miValueTuple );
		}
	}

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataDisassemble::Acknowledge( void )
{
	const CMICmnMIValueResult miValueResult( "asm_insns", m_miValueList );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
	m_miResultRecord = miRecordResult;
				
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataDisassemble::CreateSelf( void )
{
	return new CMICmdCmdDataDisassemble();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataReadMemoryBytes constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataReadMemoryBytes::CMICmdCmdDataReadMemoryBytes( void )
:	m_constStrArgThread( "thread" )
,	m_constStrArgByteOffset( "o" )	
,	m_constStrArgAddrStart( "address" )	
,	m_constStrArgNumBytes( "count" )	
,	m_pBufferMemory( nullptr )
,	m_nAddrStart( 0 )
,	m_nAddrNumBytesToRead( 0 )
,	m_nAddrOffset( 0 )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-read-memory-bytes";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataReadMemoryBytes::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataReadMemoryBytes destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataReadMemoryBytes::~CMICmdCmdDataReadMemoryBytes( void )
{
	if( m_pBufferMemory != nullptr )
	{
		delete [] m_pBufferMemory;
		m_pBufferMemory = nullptr;
	}
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataReadMemoryBytes::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThread, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgByteOffset, false, true, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValNumber( m_constStrArgAddrStart, true, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValNumber( m_constStrArgNumBytes, true, true ) ) );
	return (bOk && ParseValidateCmdOptions() );
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataReadMemoryBytes::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgAddrStart, Number, m_constStrArgAddrStart );
	CMICMDBASE_GETOPTION( pArgAddrOffset, Number, m_constStrArgByteOffset );
	CMICMDBASE_GETOPTION( pArgNumBytes, Number, m_constStrArgNumBytes );

	const MIuint64 nAddrStart = pArgAddrStart->GetValue(); 
	const MIuint64 nAddrNumBytes = pArgNumBytes->GetValue();
	if( pArgAddrOffset->GetFound() )
		m_nAddrOffset = pArgAddrOffset->GetValue();
	
	m_pBufferMemory = new MIuchar[ nAddrNumBytes ];
	if( m_pBufferMemory == nullptr )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_MEMORY_ALLOC_FAILURE ), m_cmdData.strMiCmd.c_str(), nAddrNumBytes ) );
		return MIstatus::failure;
	}
	
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	lldb::SBError error;
	const MIuint64 nReadBytes = rProcess.ReadMemory( static_cast< lldb::addr_t >( nAddrStart ), (void *) m_pBufferMemory, nAddrNumBytes, error );
	if( nReadBytes != nAddrNumBytes )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_LLDB_ERR_NOT_READ_WHOLE_BLK ), m_cmdData.strMiCmd.c_str(), nAddrNumBytes, nAddrStart ) );
		return MIstatus::failure;
	}
	if( error.Fail() )
	{
		lldb::SBStream err;
		const bool bOk = error.GetDescription( err ); MIunused( bOk );
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_LLDB_ERR_READ_MEM_BYTES ), m_cmdData.strMiCmd.c_str(), nAddrNumBytes, nAddrStart, err.GetData() ) );
		return MIstatus::failure;
	}

	m_nAddrStart = nAddrStart;
	m_nAddrNumBytesToRead = nAddrNumBytes;
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataReadMemoryBytes::Acknowledge( void )
{
	// MI: memory=[{begin=\"0x%08x\",offset=\"0x%08x\",end=\"0x%08x\",contents=\" \" }]"
	const CMICmnMIValueConst miValueConst( CMIUtilString::Format( "0x%08x", m_nAddrStart ) );
	const CMICmnMIValueResult miValueResult( "begin", miValueConst );
	CMICmnMIValueTuple miValueTuple( miValueResult );
	const CMICmnMIValueConst miValueConst2( CMIUtilString::Format( "0x%08x", m_nAddrOffset ) );
	const CMICmnMIValueResult miValueResult2( "offset", miValueConst2 );
	miValueTuple.Add( miValueResult2 ); 
	const CMICmnMIValueConst miValueConst3( CMIUtilString::Format( "0x%08x", m_nAddrStart + m_nAddrNumBytesToRead ) );
	const CMICmnMIValueResult miValueResult3( "end", miValueConst3 );
	miValueTuple.Add( miValueResult3 );

	// MI: contents=\" \"
	CMIUtilString strContent;
	strContent.reserve( (m_nAddrNumBytesToRead << 1) + 1 );
	for( MIuint64 i = 0; i < m_nAddrNumBytesToRead; i ++ )
	{
		strContent += CMIUtilString::Format( "%02x", m_pBufferMemory[ i ] );
	}
	const CMICmnMIValueConst miValueConst4( strContent );
	const CMICmnMIValueResult miValueResult4( "contents", miValueConst4 );
	miValueTuple.Add( miValueResult4 );
	const CMICmnMIValueList miValueList( miValueTuple );
	const CMICmnMIValueResult miValueResult5( "memory", miValueList );
	
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult5 );
	m_miResultRecord = miRecordResult;
				
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataReadMemoryBytes::CreateSelf( void )
{
	return new CMICmdCmdDataReadMemoryBytes();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataReadMemory constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataReadMemory::CMICmdCmdDataReadMemory( void )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-read-memory";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataReadMemory::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataReadMemory destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataReadMemory::~CMICmdCmdDataReadMemory( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataReadMemory::Execute( void )
{
	// Do nothing - command deprecated use "data-read-memory-bytes" command
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataReadMemory::Acknowledge( void )
{
	// Command CMICmdCmdSupportListFeatures sends "data-read-memory-bytes" which causes this command not to be called
	const CMICmnMIValueConst miValueConst( MIRSRC( IDS_CMD_ERR_NOT_IMPLEMENTED_DEPRECATED ) );
	const CMICmnMIValueResult miValueResult( "msg", miValueConst );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
	m_miResultRecord = miRecordResult;
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataReadMemory::CreateSelf( void )
{
	return new CMICmdCmdDataReadMemory();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataListRegisterNames constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataListRegisterNames::CMICmdCmdDataListRegisterNames( void )
:	m_constStrArgThreadGroup( "thread-group" )
,	m_constStrArgRegNo( "regno" )
,	m_miValueList( true )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-list-register-names";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataListRegisterNames::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataReadMemoryBytes destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataListRegisterNames::~CMICmdCmdDataListRegisterNames( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterNames::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThreadGroup, false, false, CMICmdArgValListBase::eArgValType_ThreadGrp, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValListOfN( m_constStrArgRegNo, false, false, CMICmdArgValListBase::eArgValType_Number ) ) );
	return (bOk && ParseValidateCmdOptions() );
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterNames::Execute( void )
{
	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	if( !rProcess.IsValid() )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_INVALID_PROCESS ), m_cmdData.strMiCmd.c_str() ) );
		return MIstatus::failure;
	}
	
	lldb::SBThread thread = rProcess.GetSelectedThread();
	lldb::SBFrame frame = thread.GetSelectedFrame();
	lldb::SBValueList registers = frame.GetRegisters();
	const MIuint nRegisters = registers.GetSize();
	for( MIuint i = 0; i < nRegisters; i++ )
	{
		lldb::SBValue value = registers.GetValueAtIndex( i );
		const MIuint nRegChildren = value.GetNumChildren();
		for( MIuint j = 0; j < nRegChildren; j++ )
		{
			lldb::SBValue value2 = value.GetChildAtIndex( j );
			if( value2.IsValid() )
			{
				const CMICmnMIValueConst miValueConst( CMICmnLLDBUtilSBValue( value2 ).GetName() );
				m_miValueList.Add( miValueConst );
			}
		}
	}
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterNames::Acknowledge( void )
{
	const CMICmnMIValueResult miValueResult( "register-names", m_miValueList );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
	m_miResultRecord = miRecordResult;
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataListRegisterNames::CreateSelf( void )
{
	return new CMICmdCmdDataListRegisterNames();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataListRegisterValues constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataListRegisterValues::CMICmdCmdDataListRegisterValues( void )
:	m_constStrArgThread( "thread" )
,	m_constStrArgSkip( "skip-unavailable" )
,	m_constStrArgFormat( "fmt" )
,	m_constStrArgRegNo( "regno" )
,	m_miValueList( true )
,	m_pProcess( nullptr )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-list-register-values";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataListRegisterValues::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataListRegisterValues destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataListRegisterValues::~CMICmdCmdDataListRegisterValues( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterValues::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThread, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgSkip, false, false ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgFormat, true, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValListOfN( m_constStrArgRegNo, false, true, CMICmdArgValListBase::eArgValType_Number ) ) );
	return (bOk && ParseValidateCmdOptions() );
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterValues::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgFormat, String, m_constStrArgFormat );
	CMICMDBASE_GETOPTION( pArgRegNo, ListOfN, m_constStrArgRegNo );
	
	const CMIUtilString & rStrFormat( pArgFormat->GetValue() );
	if( rStrFormat.length() != 1 )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_INVALID_FORMAT_TYPE ), m_cmdData.strMiCmd.c_str(), rStrFormat.c_str() ) );
		return MIstatus::failure;
	}
	const CMICmnLLDBDebugSessionInfoVarObj::varFormat_e eFormat = CMICmnLLDBDebugSessionInfoVarObj::GetVarFormatForChar( rStrFormat[ 0 ] );
	if( eFormat == CMICmnLLDBDebugSessionInfoVarObj::eVarFormat_Invalid )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_INVALID_FORMAT_TYPE ), m_cmdData.strMiCmd.c_str(), rStrFormat.c_str() ) );
		return MIstatus::failure;
	}

	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	if( !rProcess.IsValid() )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_INVALID_PROCESS ), m_cmdData.strMiCmd.c_str() ) );
		return MIstatus::failure;
	}
	m_pProcess = &rProcess;

	const CMICmdArgValListBase::VecArgObjPtr_t & rVecRegNo( pArgRegNo->GetExpectedOptions() );
	CMICmdArgValListBase::VecArgObjPtr_t::const_iterator it = rVecRegNo.begin();
	while( it != rVecRegNo.end() )
	{
		const CMICmdArgValNumber * pRegNo = static_cast< CMICmdArgValNumber * >( *it );
		const MIuint nReg = pRegNo->GetValue();
		lldb::SBValue regValue = GetRegister( nReg );
		const CMIUtilString strRegValue( CMICmnLLDBDebugSessionInfoVarObj::GetValueStringFormatted( regValue, eFormat ) );

		const CMICmnMIValueConst miValueConst( CMIUtilString::Format( "%u", nReg ) );
		const CMICmnMIValueResult miValueResult( "number", miValueConst );
		CMICmnMIValueTuple miValueTuple( miValueResult );
		const CMICmnMIValueConst miValueConst2( strRegValue );
		const CMICmnMIValueResult miValueResult2( "value", miValueConst2 );
		miValueTuple.Add( miValueResult2 );
		m_miValueList.Add( miValueTuple );

		// Next
		++it;
	}
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterValues::Acknowledge( void )
{
	const CMICmnMIValueResult miValueResult( "register-values", m_miValueList );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done, miValueResult );
	m_miResultRecord = miRecordResult;
				
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataListRegisterValues::CreateSelf( void )
{
	return new CMICmdCmdDataListRegisterValues();
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Method.
// Args:	None.
// Return:	lldb::SBValue - LLDB SBValue object.
// Throws:	None.
//--
lldb::SBValue CMICmdCmdDataListRegisterValues::GetRegister( const MIuint vRegisterIndex ) const
{
	lldb::SBThread thread = m_pProcess->GetSelectedThread();
	lldb::SBFrame frame = thread.GetSelectedFrame();
	lldb::SBValueList registers = frame.GetRegisters();
	const MIuint nRegisters = registers.GetSize();
	for( MIuint i = 0; i < nRegisters; i++ )
	{
		lldb::SBValue value = registers.GetValueAtIndex( i );
		const MIuint nRegChildren = value.GetNumChildren();
		if( nRegChildren > 0 )
		{
			lldb::SBValue value2 = value.GetChildAtIndex( vRegisterIndex );
			if( value2.IsValid() )
			{
				return value2;
			}
		}
	}

	return lldb::SBValue();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataListRegisterChanged constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataListRegisterChanged::CMICmdCmdDataListRegisterChanged( void )
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-list-changed-registers";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataListRegisterChanged::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataListRegisterChanged destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataListRegisterChanged::~CMICmdCmdDataListRegisterChanged( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterChanged::Execute( void )
{
	// Do nothing
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataListRegisterChanged::Acknowledge( void )
{
	const CMICmnMIValueConst miValueConst( MIRSRC( IDS_WORD_NOT_IMPLEMENTED ) );
	const CMICmnMIValueResult miValueResult( "msg", miValueConst );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
	m_miResultRecord = miRecordResult;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataListRegisterChanged::CreateSelf( void )
{
	return new CMICmdCmdDataListRegisterChanged();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataWriteMemoryBytes constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataWriteMemoryBytes::CMICmdCmdDataWriteMemoryBytes( void )
:	m_constStrArgThread( "thread" )
,	m_constStrArgAddr( "address" )	
,	m_constStrArgContents( "contents" )	
,	m_constStrArgCount( "count" )	
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-write-memory-bytes";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataWriteMemoryBytes::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataWriteMemoryBytes destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataWriteMemoryBytes::~CMICmdCmdDataWriteMemoryBytes( void )
{
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataWriteMemoryBytes::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThread, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgAddr, true, true, false, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgContents, true, true, true, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgCount, false, true, false, true ) ) );
	return (bOk && ParseValidateCmdOptions() );
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataWriteMemoryBytes::Execute( void )
{
	// Do nothing - not reproduceable (yet) in Eclipse
	//CMICMDBASE_GETOPTION( pArgOffset, OptionShort, m_constStrArgOffset );
	//CMICMDBASE_GETOPTION( pArgAddr, String, m_constStrArgAddr );
	//CMICMDBASE_GETOPTION( pArgNumber, String, m_constStrArgNumber );
	//CMICMDBASE_GETOPTION( pArgContents, String, m_constStrArgContents );
	// 
	// Numbers extracts as string types as they could be hex numbers
	// '&' is not recognised and so has to be removed

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataWriteMemoryBytes::Acknowledge( void )
{
	const CMICmnMIValueConst miValueConst( MIRSRC( IDS_WORD_NOT_IMPLEMENTED ) );
	const CMICmnMIValueResult miValueResult( "msg", miValueConst );
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Error, miValueResult );
	m_miResultRecord = miRecordResult;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataWriteMemoryBytes::CreateSelf( void )
{
	return new CMICmdCmdDataWriteMemoryBytes();
}

//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------
//---------------------------------------------------------------------------------------

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataWriteMemory constructor.
// Type:	Method.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataWriteMemory::CMICmdCmdDataWriteMemory( void )
:	m_constStrArgThread( "thread" )
,	m_constStrArgOffset( "o" )
,	m_constStrArgAddr( "address" )	
,	m_constStrArgD( "d" )
,	m_constStrArgNumber( "a number" )
,	m_constStrArgContents( "contents" )	
,	m_nAddr( 0 )
,	m_nCount( 0 )
,	m_pBufferMemory( nullptr )	
{
	// Command factory matches this name with that received from the stdin stream
	m_strMiCmd = "data-write-memory";
	
	// Required by the CMICmdFactory when registering *this command
	m_pSelfCreatorFn = &CMICmdCmdDataWriteMemory::CreateSelf;
}

//++ ------------------------------------------------------------------------------------
// Details:	CMICmdCmdDataWriteMemory destructor.
// Type:	Overrideable.
// Args:	None.
// Return:	None.
// Throws:	None.
//--
CMICmdCmdDataWriteMemory::~CMICmdCmdDataWriteMemory( void )
{
	if( m_pBufferMemory != nullptr )
	{
		delete [] m_pBufferMemory;
		m_pBufferMemory = nullptr;
	}
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The parses the command line options 
//			arguments to extract values for each of those arguments.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataWriteMemory::ParseArgs( void )
{
	bool bOk = m_setCmdArgs.Add( *(new CMICmdArgValOptionLong( m_constStrArgThread, false, false, CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValOptionShort( m_constStrArgOffset, false, true,  CMICmdArgValListBase::eArgValType_Number, 1 ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValNumber( m_constStrArgAddr, true, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValString( m_constStrArgD, true, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValNumber( m_constStrArgNumber, true, true ) ) );
	bOk = bOk && m_setCmdArgs.Add( *(new CMICmdArgValNumber( m_constStrArgContents, true, true ) ) );
	return (bOk && ParseValidateCmdOptions() );
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command does work in this function.
//			The command is likely to communicate with the LLDB SBDebugger in here.
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataWriteMemory::Execute( void )
{
	CMICMDBASE_GETOPTION( pArgOffset, OptionShort, m_constStrArgOffset );
	CMICMDBASE_GETOPTION( pArgAddr, Number, m_constStrArgAddr );
	CMICMDBASE_GETOPTION( pArgNumber, Number, m_constStrArgNumber );
	CMICMDBASE_GETOPTION( pArgContents, Number, m_constStrArgContents );

	MIuint nAddrOffset = 0;
	if( pArgOffset->GetFound() && !pArgOffset->GetExpectedOption< CMICmdArgValNumber, MIuint>( nAddrOffset ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ARGS_ERR_VALIDATION_INVALID ), m_cmdData.strMiCmd.c_str(), m_constStrArgAddr.c_str() ) );
		return MIstatus::failure;
	}
	m_nAddr = pArgAddr->GetValue();
	m_nCount = pArgNumber->GetValue();
	const MIuint64 nValue = pArgContents->GetValue();

	m_pBufferMemory = new MIuchar [ m_nCount ];
	if( m_pBufferMemory == nullptr )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_MEMORY_ALLOC_FAILURE ), m_cmdData.strMiCmd.c_str(), m_nCount ) );
		return MIstatus::failure;
	}
	*m_pBufferMemory = static_cast< MIchar >( nValue );

	CMICmnLLDBDebugSessionInfo & rSessionInfo( CMICmnLLDBDebugSessionInfo::Instance() );
	lldb::SBProcess & rProcess = rSessionInfo.m_lldbProcess;
	lldb::SBError error;
	lldb::addr_t addr = static_cast< lldb::addr_t >( m_nAddr + nAddrOffset );
	const size_t nBytesWritten = rProcess.WriteMemory( addr, (const void *) m_pBufferMemory, (size_t) m_nCount, error ); 
	if( nBytesWritten != static_cast< size_t >( m_nCount ) )
	{
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_LLDB_ERR_NOT_WRITE_WHOLEBLK ), m_cmdData.strMiCmd.c_str(), m_nCount, addr ) );
		return MIstatus::failure;
	}
	if( error.Fail() )
	{
		lldb::SBStream err;
		const bool bOk = error.GetDescription( err ); MIunused( bOk );
		SetError( CMIUtilString::Format( MIRSRC( IDS_CMD_ERR_LLDB_ERR_WRITE_MEM_BYTES ), m_cmdData.strMiCmd.c_str(), m_nCount, addr, err.GetData() ) );
		return MIstatus::failure;
	}
	
	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	The invoker requires this function. The command prepares a MI Record Result
//			for the work carried out in the Execute().
// Type:	Overridden.
// Args:	None.
// Return:	MIstatus::success - Functional succeeded.
//			MIstatus::failure - Functional failed.
// Throws:	None.
//--
bool CMICmdCmdDataWriteMemory::Acknowledge( void )
{
	const CMICmnMIResultRecord miRecordResult( m_cmdData.strMiCmdToken, CMICmnMIResultRecord::eResultClass_Done );
	m_miResultRecord = miRecordResult;

	return MIstatus::success;
}

//++ ------------------------------------------------------------------------------------
// Details:	Required by the CMICmdFactory when registering *this command. The factory
//			calls this function to create an instance of *this command.
// Type:	Static method.
// Args:	None.
// Return:	CMICmdBase * - Pointer to a new command.
// Throws:	None.
//--
CMICmdBase * CMICmdCmdDataWriteMemory::CreateSelf( void )
{
	return new CMICmdCmdDataWriteMemory();
}
