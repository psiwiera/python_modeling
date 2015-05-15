from py2neo import Graph, Node, Relationship

graph = Graph("http://localhost:7474/db/data/")

alice = Node("Person", name="Alice")
bob = Node("Person", name="Bob")
alice_knows_bob = Relationship(alice, "KNOWS", bob)
graph.create(alice_knows_bob)