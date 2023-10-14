Vector databases, designed to store data in vectors, streamline the way LLMs access and manage information. These databases play a pivotal role across various LLM applications. For instance, in machine learning, they house the foundational training data. In natural language processing, they’re repositories for essential vocabulary and grammar rules. And for recommendation systems, serve as reservoirs of users’ specific product and service preferences.



## Installation

Before using the `VectorStore` class, you need to have NumPy installed. If you don't have it installed already, you can install it using pip:

```bash
pip install numpy
```

## Usage

To use the `VectorStore` class, you can follow these steps:

1. Import NumPy and create an instance of the `VectorStore` class:

    ```python
    import numpy as np
    from vector_store import VectorStore

    vector_store = VectorStore()
    ```

2. Adding vectors to the store:

    To add a vector to the store, use the `add_vector` method. It takes a unique identifier and a NumPy array as input.

    ```python
    vector_id = "unique_id"
    vector = np.array([0.1, 0.2, 0.3])
    vector_store.add_vector(vector_id, vector)
    ```

3. Retrieving vectors from the store:

    To retrieve a vector from the store, use the `get_vector` method. It takes the vector's identifier as input and returns the NumPy array.

    ```python
    retrieved_vector = vector_store.get_vector(vector_id)
    ```

4. Finding similar vectors:

    To find similar vectors to a given query vector, use the `find_similar_vectors` method. It takes a query vector and an optional number of results to return. The method returns a list of (vector_id, similarity_score) tuples for the most similar vectors.

    ```python
    query_vector = np.array([0.4, 0.5, 0.6])
    similar_vectors = vector_store.find_similar_vectors(query_vector, num_results=5)
    ```

## Indexing

The `VectorStore` class uses a simple brute-force approach for indexing based on cosine similarity. When a new vector is added, the indexing structure is updated to store the similarity between the new vector and all other vectors in the store. This allows for efficient retrieval of similar vectors when using the `find_similar_vectors` method.

## Example

Here's a brief example of how to use the `VectorStore` class:

```python
import numpy as np
from vector_store import VectorStore

# Create a VectorStore instance
vector_store = VectorStore()

# Add vectors
vector_store.add_vector("vector1", np.array([0.1, 0.2, 0.3]))
vector_store.add_vector("vector2", np.array([0.4, 0.5, 0.6]))

# Retrieve a vector
retrieved_vector = vector_store.get_vector("vector1")

# Find similar vectors
query_vector = np.array([0.7, 0.8, 0.9])
similar_vectors = vector_store.find_similar_vectors(query_vector, num_results=2)
```

Please refer to the class documentation for more detailed information on each method and its parameters.

## License

This `VectorStore` class is provided under the MIT License. You can find the full license in the `LICENSE` file.