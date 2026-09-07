"""
Automated validation of HTML/Markdown cleanliness in app.py.
Ensures no unclosed divs, no orphaned closing tags, and no leaked HTML text.
"""

import ast
import os
import re
import unittest


class TestHTMLCleanliness(unittest.TestCase):

    def setUp(self):
        self.app_path = os.path.join(os.path.dirname(__file__), "..", "app.py")
        with open(self.app_path, "r", encoding="utf-8") as f:
            self.app_content = f.read()
        self.tree = ast.parse(self.app_content)

    def test_no_isolated_closing_div_markdown_calls(self):
        """Verify that st.markdown('</div>') is never called alone."""
        isolated_divs = re.findall(r'st\.markdown\(\s*["\']\s*</div>\s*["\']', self.app_content)
        self.assertEqual(len(isolated_divs), 0, f"Found {len(isolated_divs)} isolated st.markdown('</div>') calls!")

    def test_balanced_html_in_st_markdown_calls(self):
        """Verify that every st.markdown with unsafe_allow_html has balanced open and close tags."""
        unbalanced = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                func = node.func
                is_st_markdown = False
                if isinstance(func, ast.Attribute) and func.attr == "markdown":
                    if isinstance(func.value, ast.Name) and func.value.id == "st":
                        is_st_markdown = True
                    elif isinstance(func.value, ast.Attribute) and func.value.attr == "sidebar":
                        is_st_markdown = True
                
                if is_st_markdown and node.args:
                    first_arg = node.args[0]
                    text_chunks = []
                    if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                        text_chunks.append(first_arg.value)
                    elif isinstance(first_arg, ast.JoinedStr):
                        for val in first_arg.values:
                            if isinstance(val, ast.Constant) and isinstance(val.value, str):
                                text_chunks.append(val.value)
                            else:
                                text_chunks.append(" PLACEHOLDER ")
                    
                    full_text = "".join(text_chunks)
                    if "<div" in full_text or "</div>" in full_text:
                        o = len(re.findall(r'<div[\s>]', full_text, re.IGNORECASE))
                        cl = len(re.findall(r'</div>', full_text, re.IGNORECASE))
                        if o != cl:
                            unbalanced.append((node.lineno, o, cl))

        self.assertEqual(len(unbalanced), 0, f"Found unbalanced st.markdown div tags at lines: {unbalanced}")

    def test_sidebar_sample_benchmark_not_in_primary_nav(self):
        """Verify Sample Benchmark is inside an expander and not in main sidebar."""
        self.assertIn('with st.sidebar.expander("Developer / Test Benchmarks"', self.app_content)


if __name__ == "__main__":
    unittest.main()
