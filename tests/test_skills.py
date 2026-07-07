"""Tests for skill extraction and taxonomy normalization."""

from cognicv.skills import extract_skills, group_by_category
from cognicv.taxonomy import canonicalize, display_name


class TestCanonicalize:
    def test_aliases_resolve(self):
        assert canonicalize("K8s") == "kubernetes"
        assert canonicalize("golang") == "go"
        assert canonicalize("scikit learn") == "scikit-learn"
        assert canonicalize("Natural Language Processing") == "nlp"
        assert canonicalize("postgres") == "postgresql"
        assert canonicalize("amazon web services") == "aws"

    def test_display_names_resolve(self):
        assert canonicalize("Node.js") == "node.js"
        assert canonicalize("PyTorch") == "pytorch"

    def test_unknown_returns_none(self):
        assert canonicalize("underwater basket weaving") is None


class TestBasicExtraction:
    def test_simple_skills(self):
        found = extract_skills("Experienced in Python, Django and PostgreSQL.")
        assert {"python", "django", "postgresql"} <= found

    def test_aliases_map_to_canonical(self):
        found = extract_skills("Deployed with K8s on Amazon Web Services.")
        assert "kubernetes" in found
        assert "aws" in found

    def test_case_insensitive(self):
        found = extract_skills("PYTHON and pyTorch and TensorFlow")
        assert {"python", "pytorch", "tensorflow"} <= found

    def test_hyphen_space_flexible(self):
        assert "scikit-learn" in extract_skills("used scikit learn for modeling")
        assert "scikit-learn" in extract_skills("used scikit-learn for modeling")
        assert "fine-tuning" in extract_skills("LLM fine tuning experience")

    def test_special_char_skills(self):
        found = extract_skills("Languages: C++, C#, and .NET development")
        assert "c++" in found
        assert "c#" in found
        assert ".net" in found

    def test_no_substring_false_positives(self):
        # "java" must not match inside "javascript"
        found = extract_skills("JavaScript developer")
        assert "javascript" in found
        assert "java" not in found

    def test_empty_text(self):
        assert extract_skills("") == set()


class TestAmbiguousTokens:
    def test_go_in_list_context(self):
        assert "go" in extract_skills("Languages: Python, Go, Rust")
        assert "go" in extract_skills("Go, Python and Rust")
        assert "go" in extract_skills("golang microservices")
        assert "go" in extract_skills("go programming experience")

    def test_go_not_the_verb(self):
        assert "go" not in extract_skills("willing to go the extra mile")
        assert "go" not in extract_skills("I go to conferences and give talks")

    def test_r_in_list_context(self):
        assert "r" in extract_skills("Skills: Python, R, SQL")
        assert "r" in extract_skills("R programming and statistics")

    def test_r_not_the_letter(self):
        assert "r" not in extract_skills("worked with r&d teams closely")
        assert "r" not in extract_skills("part r of the document")

    def test_c_in_list_context(self):
        assert "c" in extract_skills("Proficient in C, C++, and Rust")
        assert "c" in extract_skills("c programming for embedded systems")

    def test_spring_framework_not_season(self):
        assert "spring" in extract_skills("Built services with Spring Boot")
        assert "spring" in extract_skills("Java, Spring, Hibernate")
        assert "spring" not in extract_skills("Graduated Spring 2023 with honors")
        assert "spring" not in extract_skills("internship in spring semester")

    def test_excel_tool_not_verb(self):
        assert "excel" in extract_skills("Advanced Microsoft Excel and VBA")
        assert "excel" in extract_skills("Tools: Excel, Tableau, SQL")
        assert "excel" not in extract_skills("I excel in fast-paced environments")
        assert "excel" not in extract_skills("students who excel at math")

    def test_node_framework_not_graph_node(self):
        assert "node.js" in extract_skills("Backend: Node.js and Express")
        assert "node.js" in extract_skills("nodejs microservices")
        assert "node.js" not in extract_skills("each node in the network graph")


class TestGrouping:
    def test_group_by_category(self):
        grouped = group_by_category({"python", "react", "aws"})
        assert "Python" in grouped["Programming Languages"]
        assert "React" in grouped["Frontend"]
        assert "AWS" in grouped["Cloud & DevOps"]

    def test_display_name(self):
        assert display_name("node.js") == "Node.js"
        assert display_name("c++") == "C++"
