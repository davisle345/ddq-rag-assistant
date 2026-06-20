"""
Bulk PDF Import Utility for the DDQ System
Extracts Q&A pairs from existing DDQ PDFs and adds them to the knowledge base,
skipping semantic duplicates.

Usage:
    Single file:  python bulk_pdf_import.py <pdf_file>
    Batch mode:   python bulk_pdf_import.py --batch <directory>
"""

import os
import sys
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings

import config
from rag_enhancements import BulkPDFImporter, SemanticDeduplicator
from pdf_text_extractor import PdfTextExtractor


def import_ddq_pdf(pdf_path: str, kb_path: str = config.KNOWLEDGE_BASE_PATH):
    """
    Import Q&A pairs from a DDQ PDF into the knowledge base.

    Args:
        pdf_path: Path to the DDQ PDF file
        kb_path: Path to the knowledge base CSV file

    Returns:
        Dict with import statistics
    """
    print(f"Processing PDF: {pdf_path}")

    # Initialize PDF extractor
    output_path = "temp_extracted.md"
    pdf_toolkit = PdfTextExtractor(input_path=pdf_path, output_path=output_path)

    # Initialize importer
    importer = BulkPDFImporter(pdf_toolkit)

    # Extract Q&A pairs
    print("Extracting Q&A pairs from PDF...")
    qa_pairs = importer.extract_qa_pairs(pdf_path)

    if not qa_pairs:
        print("No Q&A pairs found in PDF")
        return {'success': False, 'pairs_found': 0, 'pairs_added': 0}

    print(f"Found {len(qa_pairs)} Q&A pairs")

    # Check for duplicates
    print("Checking for semantic duplicates...")
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    deduplicator = SemanticDeduplicator(embeddings, threshold=0.90)

    # Load existing questions
    df_existing = pd.read_csv(kb_path)
    existing_questions = df_existing['Question'].tolist()

    # Filter out duplicates
    unique_pairs = []
    duplicate_count = 0

    for pair in qa_pairs:
        duplicates = deduplicator.find_duplicates(pair['question'], existing_questions)

        if duplicates:
            print(f"Duplicate found: '{pair['question'][:60]}...'")
            print(f"   Similar to: '{duplicates[0][0][:60]}...' (similarity: {duplicates[0][1]:.2%})")
            duplicate_count += 1
        else:
            unique_pairs.append(pair)

    if not unique_pairs:
        print(f"All {len(qa_pairs)} pairs were duplicates. No new data added.")
        return {
            'success': True,
            'pairs_found': len(qa_pairs),
            'pairs_added': 0,
            'duplicates_skipped': duplicate_count
        }

    # Add unique pairs to knowledge base
    print(f"Adding {len(unique_pairs)} unique pairs to knowledge base...")
    pairs_added = importer.append_to_knowledge_base(unique_pairs, kb_path)

    print(f"Successfully added {pairs_added} new Q&A pairs!")
    print("Summary:")
    print(f"   - Total pairs found: {len(qa_pairs)}")
    print(f"   - Duplicates skipped: {duplicate_count}")
    print(f"   - New pairs added: {pairs_added}")

    # Clean up temp file
    if os.path.exists(output_path):
        os.remove(output_path)

    return {
        'success': True,
        'pairs_found': len(qa_pairs),
        'pairs_added': pairs_added,
        'duplicates_skipped': duplicate_count
    }


def batch_import_pdfs(pdf_directory: str, kb_path: str = config.KNOWLEDGE_BASE_PATH):
    """
    Import Q&A pairs from all PDFs in a directory.

    Args:
        pdf_directory: Directory containing DDQ PDF files
        kb_path: Path to the knowledge base CSV file

    Returns:
        Dict with batch import statistics
    """
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith('.pdf')]

    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return {'success': False, 'files_processed': 0}

    print(f"Found {len(pdf_files)} PDF files to process")

    total_stats = {
        'files_processed': 0,
        'total_pairs_found': 0,
        'total_pairs_added': 0,
        'total_duplicates': 0,
        'failed_files': []
    }

    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_file}")
        print(f"{'='*60}")

        try:
            result = import_ddq_pdf(pdf_path, kb_path)

            if result['success']:
                total_stats['files_processed'] += 1
                total_stats['total_pairs_found'] += result['pairs_found']
                total_stats['total_pairs_added'] += result['pairs_added']
                total_stats['total_duplicates'] += result.get('duplicates_skipped', 0)
            else:
                total_stats['failed_files'].append(pdf_file)

        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")
            total_stats['failed_files'].append(pdf_file)

    print(f"\n{'='*60}")
    print("BATCH IMPORT SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {total_stats['files_processed']}/{len(pdf_files)}")
    print(f"Total Q&A pairs found: {total_stats['total_pairs_found']}")
    print(f"Total pairs added: {total_stats['total_pairs_added']}")
    print(f"Total duplicates skipped: {total_stats['total_duplicates']}")

    if total_stats['failed_files']:
        print(f"\nFailed files ({len(total_stats['failed_files'])}):")
        for failed_file in total_stats['failed_files']:
            print(f"   - {failed_file}")

    return total_stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python bulk_pdf_import.py <pdf_file>")
        print("  Batch mode:  python bulk_pdf_import.py --batch <directory>")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 3:
            print("Error: Please provide directory path for batch mode")
            sys.exit(1)
        batch_import_pdfs(sys.argv[2])
    else:
        import_ddq_pdf(sys.argv[1])
