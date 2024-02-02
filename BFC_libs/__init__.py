"""
# Libraries for Biofunctional Catalyst research team (Nakamura lab.)
##データ階層
疑似的な抽象クラスを継承した、次のような階層になる
    DataFile
        |-DataSeriese(or list[DataSeriese])
            |-ValueObjectArray(numpy.array)
                |-[ValueObject, ValueObject, ValueObject, ..., ValueObject,]
            |-ValueObjectArray  
            |-ValueObjectArray  
            ...  
"""