
install:
	pdm sync

lock:
	pdm lock

test:
	pdm test

build_doc:
	nox -s doc -R

test_in_nox_env:
	nox -s test
