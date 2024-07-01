The tests/ folder contains the test files. 

Run the full test suite with pytest by changing directory to inkstone/ (containing simulator.py) and using the command `pytest tests/`.
To run a specific test file, specify the file, e.g. `pytest tests/test_differentiable.py`.
To run a specific folder, specify the folder, e.g. `pytest tests/test_inkstone`.
To run all tests within a file containing a certain string, use the -k flag, e.g. `pytest -k "indexAssign" tests/test_in_place_operations.py` 
NOTE: Due to the relative imports located in each test file, you have to run the tests from the inkstone/ directory.

Within tests/ there are miscellaneous test files and inkstone test files. The miscellaneous test files test a variety of functionalities,
e.g. the autodiff libraries and what they can differentiate, the GenericBackend.py functions and differences between numpy and scipy.
The inkstone test files are located in tests/test_inkstone and test simulations created using the inkstone Simulator. In particular,
they test: 
    Consistency between the new, variable backends and the old, static numpy backend
    Automatic differentiability