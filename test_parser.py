#!/usr/bin/env python3
"""
Test script for PDF Parser

This script provides basic tests and validation for the PDF parser functionality.
"""

import json
import sys
from pathlib import Path
import tempfile
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdf_parser import PDFParser, ContentExtractor


def test_json_structure():
    """Test that the JSON structure is correct."""
    print("Testing JSON structure validation...")

    # Load sample output
    sample_path = Path("sample_output.json")
    if not sample_path.exists():
        print("Error: sample_output.json not found")
        return False

    try:
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate required top-level structure
        if "pages" not in data:
            print("Error: Missing 'pages' key in JSON structure")
            return False

        if not isinstance(data["pages"], list):
            print("Error: 'pages' should be a list")
            return False

        # Validate page structure
        for page in data["pages"]:
            if not isinstance(page, dict):
                print("Error: Each page should be a dictionary")
                return False

            if "page_number" not in page or "content" not in page:
                print("Error: Page missing required keys 'page_number' or 'content'")
                return False

            if not isinstance(page["content"], list):
                print("Error: Page content should be a list")
                return False

            # Validate content items
            for content_item in page["content"]:
                if not isinstance(content_item, dict):
                    print("Error: Content item should be a dictionary")
                    return False

                required_keys = ["type", "section", "subsection"]
                for key in required_keys:
                    if key not in content_item:
                        print(f"Error: Content item missing required key '{key}'")
                        return False

                # Type-specific validation
                content_type = content_item["type"]
                if content_type == "paragraph":
                    if "text" not in content_item:
                        print("Error: Paragraph missing 'text' key")
                        return False
                elif content_type == "table":
                    if "data" not in content_item:
                        print("Error: Table missing 'data' key")
                        return False
                    if not isinstance(content_item["data"], list):
                        print("Error: Table data should be a list")
                        return False
                elif content_type == "chart":
                    if "caption" not in content_item or "data" not in content_item:
                        print("Error: Chart missing 'caption' or 'data' key")
                        return False

        print("✓ JSON structure validation passed")
        return True

    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return False
    except Exception as e:
        print(f"Error: Unexpected error during validation - {e}")
        return False


def test_content_extractor():
    """Test the ContentExtractor class functionality."""
    print("Testing ContentExtractor functionality...")

    try:
        extractor = ContentExtractor()

        # Test text cleaning
        dirty_text = "  This   is   messy    text  with   spaces  "
        clean_text = extractor.clean_text(dirty_text)
        expected = "This is messy text with spaces"

        if clean_text != expected:
            print(
                f"Error: Text cleaning failed. Expected: '{expected}', Got: '{clean_text}'"
            )
            return False

        # Test heading detection
        is_heading, level = extractor.is_heading("Test Heading", 16, True, 12)
        if not is_heading:
            print("Error: Failed to detect heading")
            return False

        print("✓ ContentExtractor tests passed")
        return True

    except Exception as e:
        print(f"Error: ContentExtractor test failed - {e}")
        return False


def test_pdf_parser_initialization():
    """Test PDFParser initialization."""
    print("Testing PDFParser initialization...")

    try:
        parser = PDFParser()

        # Check that parser has required attributes
        if not hasattr(parser, "extractor"):
            print("Error: PDFParser missing extractor attribute")
            return False

        if not hasattr(parser, "logger"):
            print("Error: PDFParser missing logger attribute")
            return False

        print("✓ PDFParser initialization test passed")
        return True

    except Exception as e:
        print(f"Error: PDFParser initialization failed - {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("Running PDF Parser Test Suite")
    print("=" * 50)

    tests = [
        test_json_structure,
        test_content_extractor,
        test_pdf_parser_initialization,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Empty line for readability
        except Exception as e:
            print(f"Test failed with exception: {e}")
            print()

    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The PDF parser is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
