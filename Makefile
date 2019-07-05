.PHONY: docs help
.DEFAULT_GOAL := help

docs: ## generate Sphinx HTML documentation, including API docs
	rm -f docs/pylandstats.rst
	rm -f docs/modules.rst
	# sphinx-apidoc -o docs pylandstats
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html
