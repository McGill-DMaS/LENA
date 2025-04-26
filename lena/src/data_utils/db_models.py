from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, LargeBinary, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()

class Function(Base):
    __tablename__ = 'function'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    program = Column(Text)
    compiler = Column(String)
    optimization = Column(String)
    src = Column(Text)

    pool_pairs_view1 = relationship("PoolPairs", back_populates="view1_function", foreign_keys='PoolPairs.view1_id')
    pool_pairs_view2 = relationship("PoolPairs", back_populates="view2_function", foreign_keys='PoolPairs.view2_id')
    trains = relationship("Train", back_populates="function")
    tests = relationship("Test", back_populates="function")
    validations = relationship("Validation", back_populates="function")


class PoolPairs(Base):
    __tablename__ = 'pool_pairs'
    id = Column(Integer, primary_key=True)
    view1_id = Column(Integer, ForeignKey('function.id'))
    view2_id = Column(Integer, ForeignKey('function.id'))

    view1_function = relationship("Function", foreign_keys=[view1_id], back_populates="pool_pairs_view1")
    view2_function = relationship("Function", foreign_keys=[view2_id], back_populates="pool_pairs_view2")


class Train(Base):
    __tablename__ = 'train'
    id = Column(Integer, primary_key=True)
    function_id = Column(Integer, ForeignKey('function.id'))

    function = relationship("Function", back_populates="trains")

class Test(Base):
    __tablename__ = 'test'
    id = Column(Integer, primary_key=True)
    function_id = Column(Integer, ForeignKey('function.id'))

    function = relationship("Function", back_populates="tests")

class Validation(Base):
    __tablename__ = 'validation'
    id = Column(Integer, primary_key=True)
    function_id = Column(Integer, ForeignKey('function.id'))

    function = relationship("Function", back_populates="validations")


