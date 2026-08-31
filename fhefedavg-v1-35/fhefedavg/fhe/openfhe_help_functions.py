import openfhe as fhe

def fhe_serialize_file(obj, path, ser_type):
    """Serializes an openfhe object obj to a file stored at path.
    Uses serialization type ser_type.

    Args:
        obj (openfhe.CryptoContext | openfhe.PublicKey | openfhe.PrivateKey | openfhe.Ciphertext | openfhe.EvalKey): 
            The object to serialize.

        path (string): The path, including filename, to where the serialization should be stored.

        ser_type (openfhe.JSON | openfhe.BINARY): The type of serialization to use, either JSON or binary.

    Raises:
        Exception: Raises exception if serialization fails.
    """
    if not fhe.SerializeToFile(path, obj, ser_type):
        raise Exception(
            "Error writing serialization of to file."
        )
            
def fhe_deserialize_file(fhe_type, path, ser_type):
    """Deserializes an openfhe object with type fhe_type from the file stored
    at path. The file must be of type ser_type, either JSON or binary.

    Args:
        fhe_type (str): The type of object to deserialize. Supported types: 
            CryptoContext, 
            PublicKey, 
            PrivateKey, 
            Ciphertext, 
            EvalKey.

        path (string): The path, including filename, to where the serialization is stored.

        ser_type (openfhe.JSON | openfhe.BINARY): The type of serialization to use, 
            either JSON or binary.

    Raises:
        Exception: Raises exception if fhe_type is not a supported type.

        Exception: Raises exception if something went during deserialization.

    Returns:
        openfhe.CryptoContext | openfhe.PublicKey | 
        openfhe.PrivateKey | openfhe.Ciphertext | openfhe.EvalKey: 
            The deserialized object.
    """
    obj = None
    res = None
    match fhe_type:
        case "CryptoContext":
            obj, res = fhe.DeserializeCryptoContext(path, ser_type)
        case "PublicKey":
            obj, res = fhe.DeserializePublicKey(path, ser_type)
        case "PrivateKey":
            obj, res = fhe.DeserializePrivateKey(path, ser_type)
        case "Ciphertext":
            obj, res = fhe.DeserializeCiphertext(path, ser_type)
        case "EvalKey":
            obj, res = fhe.DeserializeEvalKey(path, ser_type)
        case _:
            raise ValueError(
                f"{fhe_type} is not a supported type."
            )

    if not res:
        raise Exception(
            "Error reading serialization from file."
        )
    
    return obj

def fhe_serialize_string(obj, ser_type):
    """Serializes an openfhe object obj to a string.
    Uses serialization type ser_type.

    Args:
        obj (openfhe.CryptoContext | openfhe.PublicKey | openfhe.PrivateKey | openfhe.Ciphertext | openfhe.EvalKey): 
            The object to serialize.

        ser_type (openfhe.JSON | openfhe.BINARY): The type of serialization to use, either JSON or binary.

    Raises:
        Exception: Raises exception if serialization fails.

    Returns:
        Str: The string serialization of the object.
    """
    serialization = fhe.Serialize(obj, ser_type)
    if not serialization:
        raise Exception(
            "Error writing serialization to string."
        )
    return serialization
            
def fhe_deserialize_string(obj_str, fhe_type, ser_type):
    """Deserializes an openfhe object with type fhe_type from the string obj_str. 
    The string serialization must be of type ser_type, either JSON or binary.

    Args:
        obj_str (str): The string to deserialize.

        fhe_type (str): The type of object to deserialize. Supported types: 
            CryptoContext, 
            PublicKey, 
            PrivateKey, 
            Ciphertext, 
            EvalKey.

        ser_type (openfhe.JSON | openfhe.BINARY): The type of serialization to use, 
            either JSON or binary.

    Returns:
        openfhe.CryptoContext | openfhe.PublicKey | 
        openfhe.PrivateKey | openfhe.Ciphertext | openfhe.EvalKey: 
            The deserialized object.
    """
    obj = None
    match fhe_type:
        case "CryptoContext":
            obj = fhe.DeserializeCryptoContextString(obj_str, ser_type)
        case "PublicKey":
            obj = fhe.DeserializePublicKeyString(obj_str, ser_type)
        case "PrivateKey":
            obj = fhe.DeserializePrivateKeyString(obj_str, ser_type)
        case "Ciphertext":
            obj = fhe.DeserializeCiphertextString(obj_str, ser_type)
        case "EvalKey":
            obj = fhe.DeserializeEvalKeyString(obj_str, ser_type)
        case _:
            raise ValueError(
                f"{fhe_type} is not a supported type."
            )
    
    return obj
