import os
import sys
from logging import INFO
import time

from flwr.common import (Context, 
                         Message, 
                         ConfigRecord, 
                         RecordDict)
from flwr.common.logger import log
from flwr.server import Grid
import openfhe as fhe

from fhefedavg.fhe.openfhe_help_functions import (fhe_deserialize_file, 
                                                  fhe_deserialize_string, 
                                                  fhe_serialize_string,
                                                  fhe_serialize_file)

def construct_cc(msg, context):
    """Construct a cryptocontext based on msg and context."""
    # Get config values from context.
    scheme = context.run_config['scheme']
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON
    securityLevel = context.run_config['security-level']
    if securityLevel == "128c":
        securityLevel = fhe.SecurityLevel.HEStd_128_classic
    elif securityLevel == "192c":
        securityLevel = fhe.SecurityLevel.HEStd_192_classic
    elif securityLevel == "256c":
        securityLevel = fhe.SecurityLevel.HEStd_256_classic
    ringDim = context.run_config['ring-dim']
    multDepth = 0
    num_clients = context.run_config['num-clients']

    # Get contents of message.
    id = msg.content.config_records['start-info']['id']

    storage = os.path.abspath('storage/') + "/"
    # If key generation has been done previosly,
    # must delete existing files.
    log(INFO, "Deleting old key files.")
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if os.path.isfile(storage+"sk"):
        os.remove(storage+"sk")
    if os.path.isfile(storage+"jpk"):
        os.remove(storage+"jpk")

    secretKeyDist = fhe.UNIFORM_TERNARY
    sigma = 3.19

    log(INFO, "Starting keygen for %s with level %d for %d nodes.", scheme, multDepth, num_clients)
    """Should find out what proper choices of parameters are.
    Both here and for master I am simply using the same 
    parameter as those used in examples."""
    if scheme == "CKKS" or scheme == "CKKS-NF":
        delta_bits = context.run_config['delta-bits']
        if delta_bits == "high":
            q_0_bits = 60
            delta_bits = 50
        elif delta_bits == "low":
            q_0_bits = 45
            delta_bits = 35

        parameters = fhe.CCParamsCKKSRNS()
        parameters.SetSecurityLevel(securityLevel)
        parameters.SetMultiplicativeDepth(multDepth)
        parameters.SetRingDim(ringDim)
        batchsize = ringDim//2 # Int division for å få heltall istedenfor .0
        parameters.SetBatchSize(batchsize) 
        parameters.SetSecretKeyDist(secretKeyDist)
        parameters.SetStandardDeviation(sigma)
        parameters.SetKeySwitchTechnique(fhe.BV)
        parameters.SetScalingModSize(delta_bits)
        parameters.SetFirstModSize(q_0_bits)

        if scheme == "CKKS-NF":
            mode = msg.content.config_records['start-info']['mode']
            if mode == "estimate":
                # cc will be used to generate noise estimate.
                parameters.SetExecutionMode(fhe.EXEC_NOISE_ESTIMATION)
            elif mode == "evaluation":
                # Noise estimate er såpass stort at jeg må 
                # bruke multDepth = 1 for å ikke få overflow.
                multDepth = 1
                parameters.SetMultiplicativeDepth(multDepth)
                # cc will be used for actual computations.
                parameters.SetExecutionMode(fhe.EXEC_EVALUATION)
                # fetch noise estimate from client state.
                noise_estimate = context.state.config_records['noise-estimate']['noise-estimate']
                parameters.SetNoiseEstimate(noise_estimate)
                # COMPOSITESCALINGAUTO gjør at jeg slipper å bruke NATIVEINT 128
                # for at decryption skal fungere.
                parameters.SetScalingTechnique(fhe.COMPOSITESCALINGAUTO)


    elif scheme == "BFV":
        plaintextMod = 2*num_clients + 1
        multTech = context.run_config['mult-tech']

        parameters = fhe.CCParamsBFVRNS()
        if multTech == "BEHZ":
            # BEHZ multiplication er ikke påvirket av bug, så den kan bruke lavere Q_bits.
            Q_bits = 30
            parameters.SetMultiplicationTechnique(fhe.BEHZ)
        else:
            # Alle HPS varianter er påvirket av bug som gjør at Q_bits må være 50 for 
            # at decryption skal bli riktig.
            Q_bits = 50

        parameters.SetSecurityLevel(securityLevel)
        parameters.SetMultiplicativeDepth(multDepth)
        parameters.SetRingDim(ringDim)
        parameters.SetBatchSize(ringDim)
        parameters.SetStandardDeviation(sigma)
        parameters.SetSecretKeyDist(secretKeyDist)
        parameters.SetPlaintextModulus(plaintextMod)
        parameters.SetKeySwitchTechnique(fhe.BV)
        parameters.SetScalingModSize(Q_bits)
        parameters.SetThresholdNumOfParties(num_clients) # Number of nodes involved.
        parameters.SetEvalAddCount(num_clients-1) # Maximum number of additions.
        parameters.SetKeySwitchCount(0) # Number of key switches.
        """NOISE_FLOODING_MULTIPARTY adds extra noise to the ciphertext before decrypting
        and is most secure mode of threshold FHE for BFV and BGV."""
        parameters.SetMultipartyMode(fhe.NOISE_FLOODING_MULTIPARTY)
    
    elif scheme == "BGV":
        modulusSwitchTech = fhe.FIXEDMANUAL
        plaintextMod = 2*num_clients + 1
        if num_clients == 3:
            if ringDim == 2**13:
                q_0_bits = 20
            elif ringDim == 2**14:
                q_0_bits = 21
            elif ringDim == 2**17:
                q_0_bits = 26
        elif num_clients == 5:
            if ringDim == 2**13:
                q_0_bits = 22
            elif ringDim == 2**14:
                q_0_bits = 22
            elif ringDim == 2**17:
                q_0_bits = 23
        elif num_clients == 7:
            if ringDim == 2**13:
                q_0_bits = 20
            elif ringDim == 2**14:
                q_0_bits = 23
            elif ringDim == 2**17:
                q_0_bits = 26

        # Bruk polynomial encoding ja/nei.
        no_polynomial_encoding = context.run_config['no-polynomial-encoding']
        if no_polynomial_encoding:
            # Bruk 60 bits for sikkerhets skyld. Testing på BFV og CKKS
            # + kommentar på openfhe forum viser at det ikke har noe å si så lenge
            # ring dimension forblir den samme.
            q_0_bits = 60
            # Hver client kan kryptere maksimal verdi samme som SecAgg,
            # og plaintext modulus må dermed være stor nok til å takle slike tall fra hver client.
            # Hvert tall må være innenfor (-t/2, t/2].
            if num_clients == 3:
                plaintextMod = 3*2*12000 + 1
            if num_clients == 5:
                plaintextMod = 5*2*7200 + 1
            if num_clients == 7:
                plaintextMod = 7*2*5143 + 1
       
        parameters = fhe.CCParamsBGVRNS()
        parameters.SetSecurityLevel(securityLevel)
        parameters.SetMultiplicativeDepth(multDepth)
        parameters.SetRingDim(ringDim)
        parameters.SetBatchSize(ringDim)
        parameters.SetStandardDeviation(sigma)
        parameters.SetSecretKeyDist(secretKeyDist)
        parameters.SetPlaintextModulus(plaintextMod)
        parameters.SetScalingTechnique(modulusSwitchTech)
        parameters.SetFirstModSize(q_0_bits)
        parameters.SetScalingModSize(0)
        parameters.SetKeySwitchCount(0) # Number of key switches.
        parameters.SetEvalAddCount(num_clients-1) # Maximum number of additions.
        parameters.SetKeySwitchTechnique(fhe.BV)
        parameters.SetThresholdNumOfParties(num_clients) # Number of nodes involved.
        """NOISE_FLOODING_MULTIPARTY adds extra noise to the ciphertext before decrypting
        and is most secure mode of threshold FHE for BFV and BGV."""
        parameters.SetMultipartyMode(fhe.NOISE_FLOODING_MULTIPARTY)

    else:
        raise Exception(f"{scheme} is not a valid scheme. Valid choices are BGV, BFV and CKKS.")

    # Generate cryptcontext with all parameters.
    cc = fhe.GenCryptoContext(parameters)    

    # Enable features you wish to use
    cc.Enable(fhe.PKE)
    cc.Enable(fhe.KEYSWITCH)
    cc.Enable(fhe.LEVELEDSHE)
    cc.Enable(fhe.ADVANCEDSHE)
    cc.Enable(fhe.MULTIPARTY)

    return cc


def start_forward_keygen_base(msg: Message, context: Context):
    """Starts key generation by creating the crypto context (cc)
    for the scheme requested by the server.
    Also creates the first key pair and evaluation key,
    and saves these.
    The public key (pk) and evaluation key (ek) are sent
    back to the server."""

    # Get contents of message.
    id = msg.content.config_records['start-info']['id']

    storage = os.path.abspath('storage/') + "/"

    # Construct cc.
    cc = construct_cc(msg, context)

    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Crypto context is now generated.
    kp = cc.KeyGen()
    if not kp.good():
        sys.exit("Key generation failed for 1st keypair.")
    log(INFO, f"Path to sk serialization is {storage}sk")
    log(INFO, "Key pair generated. Starting serialization.")

    # Serialize cc and own sk.
    fhe_serialize_file(cc, storage + "cc", ST)
    fhe_serialize_file(kp.secretKey, storage + "sk", ST)
    log(INFO, "Serialized cc and secret key.")

    # Generate first part of relinearization key.
    own_ek = cc.KeySwitchGen(kp.secretKey, kp.secretKey)
    log(INFO, "Generated own share of evaluation key.")

    # Send cc, own pk and own ek to server.
    cc_str = fhe_serialize_string(cc, ST)
    pk_str = fhe_serialize_string(kp.publicKey, ST)
    ek_str = fhe_serialize_string(own_ek, ST)
    config = {"cc": cc_str, "pk": pk_str, "ek": ek_str}
    reply_content = RecordDict(records={'key-dict': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")
    return reply

def start_forward_keygen_noek_base(msg: Message, context: Context):
    """Starts key generation by creating the crypto context (cc)
    for the scheme requested by the server.
    This version does not create an evaluation key!"""

    # Get contents of message.
    id = msg.content.config_records['start-info']['id']

    storage = os.path.abspath('storage/') + "/"

    # Construct cc.
    cc = construct_cc(msg, context)

    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Crypto context is now generated.
    kp = cc.KeyGen()
    if not kp.good():
        sys.exit("Key generation failed for 1st keypair.")
    log(INFO, f"Path to sk serialization is {storage}sk")
    log(INFO, "Key pair generated. Starting serialization.")

    # Serialize cc and own sk.
    fhe_serialize_file(cc, storage + "cc", ST)
    fhe_serialize_file(kp.secretKey, storage + "sk", ST)
    log(INFO, "Serialized cc and secret key.")

    # Send cc, own pk and own ek to server.
    cc_str = fhe_serialize_string(cc, ST)
    pk_str = fhe_serialize_string(kp.publicKey, ST)
    config = {"cc": cc_str, "pk": pk_str}
    reply_content = RecordDict(records={'key-dict': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")
    return reply


def continue_forward_keygen_base(msg: Message, context: Context):
    """Receives and serializes cc.
    Receives previous client's pk and uses
    it to create own key pair.
    Receives previous client's and client 0's ek
    and uses it to create own ek share.
    
    Returns own pk and ek share to server."""

    # Get config values from context.
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON
    
    # Get contents of message.
    id = msg.content.config_records['key-info']['id']
    cc_str = msg.content.config_records['key-info']['cc']
    prev_pk_str = msg.content.config_records['key-info']['pk']
    sum_ek_str = msg.content.config_records['key-info']['sum_ek']
    org_ek_str = msg.content.config_records['key-info']['org_ek']

    storage = os.path.abspath('storage/') + "/"
    # If key generation has been done previosly,
    # must delete existing files.
    log(INFO, "Deleting old key files.")
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if os.path.isfile(storage+"sk"):
        os.remove(storage+"sk")
    if os.path.isfile(storage+"jpk"):
        os.remove(storage+"jpk")

    # Convert strings to openfhe objects.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    prev_pk = fhe_deserialize_string(prev_pk_str, "PublicKey", ST)
    sum_ek = fhe_deserialize_string(sum_ek_str, "EvalKey", ST)
    org_ek = fhe_deserialize_string(org_ek_str, "EvalKey", ST)

    # Create own keypair.
    log(INFO, f"Creating key pair for node {id}.")
    kp = cc.MultipartyKeyGen(prev_pk)
    if not kp.good():
        raise Exception(f"Key generation for {id}(th/nd/rd) node failed.")
    pk = kp.publicKey
    sk = kp.secretKey

    # Generate evalMultKey{i} and evalMult{0+...+i}
    log(INFO, f"Creating evaluation key share for node {id}.")
    own_ek = cc.MultiKeySwitchGen(sk, sk, org_ek)
    own_sum_ek = cc.MultiAddEvalKeys(sum_ek, own_ek, pk.GetKeyTag())

    # Serialize cc and own sk.
    fhe_serialize_file(cc, storage+"cc", ST)
    fhe_serialize_file(sk, storage+"sk", ST)

    # Return own pk and own_sum_ek to server.
    pk_str = fhe_serialize_string(pk, ST)
    ek_str = fhe_serialize_string(own_sum_ek, ST)
    config = {"pk": pk_str, "ek": ek_str}
    reply_content = RecordDict(records={'key-dict': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")
    return reply

def continue_forward_keygen_noek_base(msg: Message, context: Context):
    """Receives and serializes cc.
    Receives previous client's pk and uses
    it to create own key pair.
    
    Returns own pk share to server.
    
    Does not create evaluation key!"""

    # Get config values from context.
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON
    
    # Get contents of message.
    id = msg.content.config_records['key-info']['id']
    cc_str = msg.content.config_records['key-info']['cc']
    prev_pk_str = msg.content.config_records['key-info']['pk']

    storage = os.path.abspath('storage/') + "/"
    # If key generation has been done previosly,
    # must delete existing files.
    log(INFO, "Deleting old key files.")
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if os.path.isfile(storage+"sk"):
        os.remove(storage+"sk")
    if os.path.isfile(storage+"jpk"):
        os.remove(storage+"jpk")

    # Convert strings to openfhe objects.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    prev_pk = fhe_deserialize_string(prev_pk_str, "PublicKey", ST)

    # Create own keypair.
    log(INFO, f"Creating key pair for node {id}.")
    kp = cc.MultipartyKeyGen(prev_pk)
    if not kp.good():
        raise Exception(f"Key generation for {id}(th/nd/rd) node failed.")
    pk = kp.publicKey
    sk = kp.secretKey

    # Serialize cc and own sk.
    fhe_serialize_file(cc, storage+"cc", ST)
    fhe_serialize_file(sk, storage+"sk", ST)

    # Return own pk and own_sum_ek to server.
    pk_str = fhe_serialize_string(pk, ST)
    config = {"pk": pk_str}
    reply_content = RecordDict(records={'key-dict': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")
    return reply


def end_forward_keygen_base(msg: Message, context: Context):
    """Last client receives pk and ek shares sum of previous client.
    It creates its own key pair, where the pk is also the 
    joint public key (jpk) to be used for all encryption.
    Also finishes the ek sum such that it is of the form
    (ek_0 + ek_1 + ... + ek_{n-1}),
    and computes its own ek product
    ek_{n-1}*(ek_0 + ek_1 + ... + ek_{n-1}).
    
    The jpk, ek sum and ek product are returned to server."""

    # Get config values from context.
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON
    
    # Get contents of message.
    id = msg.content.config_records['key-info']['id']
    cc_str = msg.content.config_records['key-info']['cc']
    prev_pk_str = msg.content.config_records['key-info']['pk']
    sum_ek_str = msg.content.config_records['key-info']['sum_ek']
    org_ek_str = msg.content.config_records['key-info']['org_ek']

    storage = os.path.abspath('storage/') + "/"
    # If key generation has been done previosly,
    # must delete existing files.
    log(INFO, "Deleting old key files.")
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if os.path.isfile(storage+"sk"):
        os.remove(storage+"sk")
    if os.path.isfile(storage+"jpk"):
        os.remove(storage+"jpk")

    # Convert strings to openfhe objects.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    prev_pk = fhe_deserialize_string(prev_pk_str, "PublicKey", ST)
    sum_ek = fhe_deserialize_string(sum_ek_str, "EvalKey", ST)
    org_ek = fhe_deserialize_string(org_ek_str, "EvalKey", ST)

    # Create own keypair.
    log(INFO, f"Creating key pair for node {id}. jpk is now generated.")
    kp = cc.MultipartyKeyGen(prev_pk)
    if not kp.good():
        raise Exception(f"Key generation for {id}(th/nd/rd) node failed.")
    jpk = kp.publicKey
    sk = kp.secretKey
    # Generate ek_{n-1} and (ek_0 + ek_1 + ... + ek_{n-1})
    log(INFO, f"Creating evaluation key share for node {id}.")
    own_ek = cc.MultiKeySwitchGen(sk, sk, org_ek)
    own_sum_ek = cc.MultiAddEvalKeys(sum_ek, own_ek, jpk.GetKeyTag())

    # Create ek_{n-1}*(ek_0 + ek_1 + ... + ek_{n-1})
    own_prod_ek = cc.MultiMultEvalKey(sk, own_sum_ek, jpk.GetKeyTag())

    # Serialize cc, jpk and own sk.
    fhe_serialize_file(cc, storage+"cc", ST)
    fhe_serialize_file(jpk, storage+"jpk", ST)
    fhe_serialize_file(sk, storage+"sk", ST)

    # Return jpk, sum_ek and prod_ek to server.
    jpk_str = fhe_serialize_string(jpk, ST)
    sum_ek_str = fhe_serialize_string(own_sum_ek, ST)
    prod_ek_str = fhe_serialize_string(own_prod_ek, ST)
    config = {"jpk": jpk_str, 
              "sum_ek": sum_ek_str, 
              "prod_ek": prod_ek_str}
    reply_content = RecordDict(records={'key-dict': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)

    log(INFO, f"Node {id} sending reply to server.")
    return reply

def end_forward_keygen_noek_base(msg: Message, context: Context):
    """Last client receives pk and ek shares sum of previous client.
    It creates its own key pair, where the pk is also the 
    joint public key (jpk) to be used for all encryption.

    The jpk is returned to server.
    
    Does not create evaluation key!"""

    # Get config values from context.
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON
    
    # Get contents of message.
    id = msg.content.config_records['key-info']['id']
    cc_str = msg.content.config_records['key-info']['cc']
    prev_pk_str = msg.content.config_records['key-info']['pk']

    storage = os.path.abspath('storage/') + "/"
    # If key generation has been done previosly,
    # must delete existing files.
    log(INFO, "Deleting old key files.")
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if os.path.isfile(storage+"sk"):
        os.remove(storage+"sk")
    if os.path.isfile(storage+"jpk"):
        os.remove(storage+"jpk")

    # Convert strings to openfhe objects.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    prev_pk = fhe_deserialize_string(prev_pk_str, "PublicKey", ST)

    # Create own keypair.
    log(INFO, f"Creating key pair for node {id}. jpk is now generated.")
    kp = cc.MultipartyKeyGen(prev_pk)
    if not kp.good():
        raise Exception(f"Key generation for {id}(th/nd/rd) node failed.")
    jpk = kp.publicKey
    sk = kp.secretKey

    # Serialize cc, jpk and own sk.
    fhe_serialize_file(cc, storage+"cc", ST)
    fhe_serialize_file(jpk, storage+"jpk", ST)
    fhe_serialize_file(sk, storage+"sk", ST)

    # Return jpk, sum_ek and prod_ek to server.
    jpk_str = fhe_serialize_string(jpk, ST)
    config = {"jpk": jpk_str}
    reply_content = RecordDict(records={'key-dict': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)

    log(INFO, f"Node {id} sending reply to server.")
    return reply


def continue_backward_keygen_base(msg: Message, context: Context):
    """Client receives jpk to serialize,
    and sum of eks and product of eks to further compute evaluation key.
    Client i receives 
    (ek_{i+1} + ... + ek_{n-1})*(ek_0 + ek_1 + ... + ek_{n-1})
    and computes
    (ek_i + ek_{i+1} + ... + ek_{n-1})*(ek_0 + ek_1 + ... + ek_{n-1}),
    which is returned to server."""

    # Get config values from context.
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Get contents of message.
    id = msg.content.config_records['key-info']['id']
    jpk_str = msg.content.config_records['key-info']['jpk']
    org_sum_ek_str = msg.content.config_records['key-info']['org_sum_ek']
    prev_sum_ek_str = msg.content.config_records['key-info']['prev_sum_ek']

    storage = os.path.abspath('storage/') + "/"

    # Convert strings to openfhe objects.
    jpk = fhe_deserialize_string(jpk_str, "PublicKey", ST)
    org_sum_ek = fhe_deserialize_string(org_sum_ek_str, "EvalKey", ST)
    prev_sum_ek = fhe_deserialize_string(prev_sum_ek_str, "EvalKey", ST)
    
    # Serialize jpk.
    log(INFO, f"Distributing jpk to node {id}.")
    fhe_serialize_file(jpk, storage+"jpk", ST)

    # Load cc and sk from files.
    cc = fhe_deserialize_file("CryptoContext", storage+"cc", ST)
    sk = fhe_deserialize_file("PrivateKey", storage+"sk", ST)

    # Compute own ek product.
    # ek_i*(ek_0 + ... + ek_{n-1})
    own_prod_ek = cc.MultiMultEvalKey(sk, org_sum_ek, jpk.GetKeyTag())

    # Compute new ek sum.
    # (ek_i + ek_{i+1} + ... + ek_{n-1})*(ek_0 + ... + ek_{n-1})
    new_sum_ek = cc.MultiAddEvalMultKeys(own_prod_ek,
                                  prev_sum_ek,  
                                  jpk.GetKeyTag())
    log(INFO, f"Node {id} contributing to evaluation key generation.")

    # Return new ek sum to server.
    new_ek_str = fhe_serialize_string(new_sum_ek, ST)
    config = {"new_sum_ek": new_ek_str}
    reply_content = RecordDict(records={'key-dict': ConfigRecord(config)})

    reply = Message(content=reply_content,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")
    return reply

def continue_backward_keygen_noek_base(msg: Message, context: Context):
    """Client receives jpk to serialize.
    
    Does not create evalaution key!"""

    # Get config values from context.
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    elif st == "JSON":
        ST = fhe.JSON

    # Get contents of message.
    id = msg.content.config_records['key-info']['id']
    jpk_str = msg.content.config_records['key-info']['jpk']

    storage = os.path.abspath('storage/') + "/"

    # Convert strings to openfhe objects.
    jpk = fhe_deserialize_string(jpk_str, "PublicKey", ST)
    
    # Serialize jpk.
    log(INFO, f"Distributing jpk to node {id}.")
    fhe_serialize_file(jpk, storage+"jpk", ST)

    reply_content = RecordDict()

    reply = Message(content=reply_content,
                    reply_to=msg)
    log(INFO, f"Node {id} sending reply to server.")
    return reply

def server_keygen(grid: Grid, context: Context) -> None:
    """OpenFHE parameters"""
    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    else:
        ST = fhe.JSON
    # Scheme to use
    SCHEME = context.run_config['scheme']
    # Level for scheme.
    MULT_DEPTH = context.run_config['multiplicative-depth']
    """Should probably make ST, SCHEME and MULT_DEPTH
    choosable by user, both here and for master."""
    securityLevel = context.run_config['security-level']


    log(INFO, "")  # Add newline for log readability
    log(INFO, "Starting key generation.")

    """This setup assumes that there exists 
    a directory 'storage/key_store'
    with a subdirectory 'server'"""
    log(INFO, "Deleting old key files.")
    storage = os.path.abspath('storage/') + "/"
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if(os.path.isfile(storage+"jek")):
        os.remove(storage+"jek")

    # Loop and wait until enough nodes are available.
    # Need every node available, since each must participate in keygen.
    min_nodes = context.run_config['num-clients']
    node_ids = []
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(2)
    
    n = len(node_ids)

    log(INFO, "All nodes have connected.")

    """Pick an arbitrary client to start key generation.
    An obvious choice is the first client in node_ids.
    This client will get an initial message to start key generation."""
    recordset = RecordDict()
    cc_record = ConfigRecord({'id': 0})
    recordset['start-info'] = cc_record
    message = Message(
        content=recordset,
        message_type="query.start_forward_keygen",  
        # target 'start_forward_keygen' method in ClientApp
        dst_node_id=node_ids[0],
        group_id="start",
    )
    log(INFO, f"Sending message to node 0 to begin key generation.")
    log(INFO, f"Parameters are: \n" 
                f"\t\tscheme = {SCHEME}, \n"
                f"\t\tnum-clients = {min_nodes}, \n"
                f"\t\tserialzation type = {ST}, \n"
                f"\t\tmultdepth = {MULT_DEPTH}, \n"
                f"\t\tsecurityLevel = {securityLevel}.")
    # The server can send message and wait for reply.
    replies = grid.send_and_receive([message])
    
    """The reply will contain the crypto context (cc), 
    that client's public key (pk) share,
    and the client's evaluation key (ek) share.
    """
    log(INFO, "Received reply.")
    reply = replies[0]
    cc_str = reply.content.config_records["key-dict"]['cc']
    pk_str = reply.content.config_records["key-dict"]['pk']
    ek_str = reply.content.config_records["key-dict"]['ek']

    # Server can save cc to file.
    # Convert to object, and then to file.
    # Should check sometime if you can go directly to file,
    # and whether that is faster.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    fhe_serialize_file(cc, storage+"cc", ST)

    """From here, server should use send_and_receive() to communicate
    with node_ids[1], ..., node_ids[n-2] 
    (assuming indexing from 0 to n-1 for number of clients n).
    Each client will receive cc, pk from previous client,
    sum of ek shares from previous clients and ek from client 0.
    I.e. the ek sum sent to client i is of the form
    (ek_0 + ek_1 + ... + ek_{i-1})
    In return, the server will get the clients pk and the ek sum
    (ek_0 + ek_1 + ... + ek_{i-1} + ek_i) from client i."""
    pk_list = [pk_str]
    ek_list = [ek_str]
    for i in range(1, n-1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                   'cc': cc_str,
                                   'pk': pk_list[i-1],
                                   'sum_ek': ek_list[i-1],
                                   'org_ek': ek_str})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_forward_keygen",  
            # target 'continue_forward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        # The server can send message and wait for reply.
        log(INFO, f"Sending message to node {i} to further generate "
            + "key pairs and evaluation key.")
        replies = grid.send_and_receive([message])
        log(INFO, "Received reply.")
        reply = replies[0]
        pk_list.append(reply.content.config_records["key-dict"]['pk'])
        ek_list.append(reply.content.config_records["key-dict"]['ek'])

    """The final client will receive the same parts
    as all the others, but the reply will be different.
    The last pk that the server receives is the joint public key (jpk).
    The server will also receive the sum of all ek shares (sum_ek),
    (ek_0 + ek_1 + ... + ek_{n-1}) and the ek product (prod_ek)
    ek_{n-1}*(ek_0 + ek_1 + ... + ek_{n-1}).
    """
    recordset = RecordDict()
    key_record = ConfigRecord({'id': n-1,
                               'cc': cc_str,
                               'pk': pk_list[-1],
                               'sum_ek': ek_list[-1],
                               'org_ek': ek_str})
    recordset['key-info'] = key_record
    message = Message(
        content=recordset,
        message_type="query.end_forward_keygen",  
        # target 'end_forward_keygen' method in ClientApp
        dst_node_id=node_ids[n-1],
        group_id="cont",
    )
    # The server can send message and wait for reply.
    log(INFO, f"Sending message to node {n-1} to generate "
        + "joint public key and continue generation of evaluation key.")
    replies = grid.send_and_receive([message])
    reply = replies[0]
    jpk_str = reply.content.config_records["key-dict"]['jpk']
    sum_ek_str = reply.content.config_records["key-dict"]['sum_ek']
    prod_ek_str = reply.content.config_records["key-dict"]['prod_ek']

    """Now the backwards part must be done
    to distribute the jpk and compute
    the joint evaluation key (jek).
    
    Each client (n-2, n-3, ..., 0)
    should receive the jpk, the sum_ek
    and the prod_ek from the previous client.
    The clients should reply with their new
    prod_ek.
    I.e., client i replies with
    (ek_i + ek_{i+1} + ... + ek_{n-1})*(ek_0 + ... + ek_{n-1}).
    
    The reply from client 0 is the jek."""
    sum_ek_list = [prod_ek_str]
    for i in range(n-2, -1, -1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                   'jpk': jpk_str,
                                   'org_sum_ek': sum_ek_str,
                                   'prev_sum_ek': sum_ek_list[n-2-i]})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_backward_keygen",  
            # target 'continue_backward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        # The server can send message and wait for reply.
        log(INFO, f"Sending message to node {i} to distribute jpk "
            + "and finish generating evaluation key.")
        replies = grid.send_and_receive([message])
        reply = replies[0]
        sum_ek_list.append(reply.content.config_records["key-dict"]['new_sum_ek'])

    # At this point, sum_ek_list[n-1] contains the jek.
    jek_str = sum_ek_list[-1]

    # Server saves jek to file.
    # Convert to object, then to file.
    log(INFO, "Serializing evaluation key.")
    jek = fhe_deserialize_string(jek_str, "EvalKey", ST)
    fhe_serialize_file(jek, storage+"jek", ST)

    log(INFO, "Key generation finished!")
    """Assuming that the clients do their computations
    before encrypting the results,
    only the server actually needs the evaluation key.
    Thus, it does not need to be distributed."""

def server_keygen_noek(grid: Grid, context: Context) -> None:
    """Version of server keygen which does not create evaluation key."""

    """OpenFHE parameters"""
    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    else:
        ST = fhe.JSON
    # Scheme to use
    SCHEME = context.run_config['scheme']
    # Level for scheme.
    MULT_DEPTH = context.run_config['multiplicative-depth']
    """Should probably make ST, SCHEME and MULT_DEPTH
    choosable by user, both here and for master."""
    securityLevel = context.run_config['security-level']


    log(INFO, "")  # Add newline for log readability
    log(INFO, "Starting key generation.")

    """This setup assumes that there exists 
    a directory 'storage/key_store'
    with a subdirectory 'server'"""
    log(INFO, "Deleting old key files.")
    storage = os.path.abspath('storage/') + "/"
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if(os.path.isfile(storage+"jek")):
        os.remove(storage+"jek")

    # Loop and wait until enough nodes are available.
    # Need every node available, since each must participate in keygen.
    min_nodes = context.run_config['num-clients']
    node_ids = []
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(2)
    
    n = len(node_ids)

    log(INFO, "All nodes have connected.")

    """Pick an arbitrary client to start key generation.
    An obvious choice is the first client in node_ids.
    This client will get an initial message to start key generation."""
    recordset = RecordDict()
    cc_record = ConfigRecord({'id': 0})
    recordset['start-info'] = cc_record
    message = Message(
        content=recordset,
        message_type="query.start_forward_keygen",  
        # target 'start_forward_keygen' method in ClientApp
        dst_node_id=node_ids[0],
        group_id="start",
    )
    log(INFO, f"Sending message to node 0 to begin key generation.")
    log(INFO, f"Parameters are: \n" 
                f"\t\tscheme = {SCHEME}, \n"
                f"\t\tnum-clients = {min_nodes}, \n"
                f"\t\tserialzation type = {ST}, \n"
                f"\t\tmultdepth = {MULT_DEPTH}, \n"
                f"\t\tsecurityLevel = {securityLevel}.")
    # The server can send message and wait for reply.
    replies = grid.send_and_receive([message])
    
    """The reply will contain the crypto context (cc) and 
    that client's public key (pk) share..
    """
    log(INFO, "Received reply.")
    reply = replies[0]
    cc_str = reply.content.config_records["key-dict"]['cc']
    pk_str = reply.content.config_records["key-dict"]['pk']

    # Server can save cc to file.
    # Convert to object, and then to file.
    # Should check sometime if you can go directly to file,
    # and whether that is faster.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    fhe_serialize_file(cc, storage+"cc", ST)

    """From here, server should use send_and_receive() to communicate
    with node_ids[1], ..., node_ids[n-2] 
    (assuming indexing from 0 to n-1 for number of clients n).
    Each client will receive cc and pk share from previous client."""
    pk_list = [pk_str]
    for i in range(1, n-1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                   'cc': cc_str,
                                   'pk': pk_list[i-1]})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_forward_keygen",  
            # target 'continue_forward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        # The server can send message and wait for reply.
        log(INFO, f"Sending message to node {i} to further generate "
            + "key pairs and evaluation key.")
        replies = grid.send_and_receive([message])
        log(INFO, "Received reply.")
        reply = replies[0]
        pk_list.append(reply.content.config_records["key-dict"]['pk'])

    """The final client will receive the same parts
    as all the others, but the reply will be different.
    The last pk that the server receives is the joint public key (jpk).
    """
    recordset = RecordDict()
    key_record = ConfigRecord({'id': n-1,
                               'cc': cc_str,
                               'pk': pk_list[-1]})
    recordset['key-info'] = key_record
    message = Message(
        content=recordset,
        message_type="query.end_forward_keygen",  
        # target 'end_forward_keygen' method in ClientApp
        dst_node_id=node_ids[n-1],
        group_id="cont",
    )
    # The server can send message and wait for reply.
    log(INFO, f"Sending message to node {n-1} to generate "
        + "joint public key and continue generation of evaluation key.")
    replies = grid.send_and_receive([message])
    reply = replies[0]
    jpk_str = reply.content.config_records["key-dict"]['jpk']

    """Now the backwards part must be done
    to distribute the jpk.
    
    Each client (n-2, n-3, ..., 0)
    should receive the jpk."""
    messages = []
    for i in range(0, n-1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                    'jpk': jpk_str})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_backward_keygen",  
            # target 'continue_backward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        messages.append(message)
    # The server can send message and wait for reply.
    log(INFO, f"Sending message to node {i} to distribute jpk "
        + "and finish generating evaluation key.")
    grid.send_and_receive(messages)

    log(INFO, "Key generation finished!")
    """Assuming that the clients do their computations
    before encrypting the results,
    only the server actually needs the evaluation key.
    Thus, it does not need to be distributed."""


def server_keygen_noise_flooding(grid: Grid, context: Context, mode: str) -> None:
    """OpenFHE parameters"""
    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    else:
        ST = fhe.JSON
    # Scheme to use
    SCHEME = context.run_config['scheme']
    # Level for scheme.
    MULT_DEPTH = context.run_config['multiplicative-depth']
    """Should probably make ST, SCHEME and MULT_DEPTH
    choosable by user, both here and for master."""
    securityLevel = context.run_config['security-level']


    log(INFO, "")  # Add newline for log readability
    log(INFO, "Starting key generation.")

    """This setup assumes that there exists 
    a directory 'storage/key_store'
    with a subdirectory 'server'"""
    log(INFO, "Deleting old key files.")
    storage = os.path.abspath('storage/') + "/"
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if(os.path.isfile(storage+"jek")):
        os.remove(storage+"jek")

    # Loop and wait until enough nodes are available.
    # Need every node available, since each must participate in keygen.
    min_nodes = context.run_config['num-clients']
    node_ids = []
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(2)
    
    n = len(node_ids)

    log(INFO, "All nodes have connected.")

    """Pick an arbitrary client to start key generation.
    An obvious choice is the first client in node_ids.
    This client will get an initial message to start key generation."""
    recordset = RecordDict()
    cc_record = ConfigRecord({'id': 0, 'mode': mode})
    recordset['start-info'] = cc_record
    message = Message(
        content=recordset,
        message_type="query.start_forward_keygen",  
        # target 'start_forward_keygen' method in ClientApp
        dst_node_id=node_ids[0],
        group_id="start",
    )
    log(INFO, f"Sending message to node 0 to begin key generation.")
    log(INFO, f"Parameters are: \n" 
                f"\t\tscheme = {SCHEME}, \n"
                f"\t\tnum-clients = {min_nodes}, \n"
                f"\t\tserialzation type = {ST}, \n"
                f"\t\tmultdepth = {MULT_DEPTH}, \n"
                f"\t\tsecurityLevel = {securityLevel}.")
    # The server can send message and wait for reply.
    replies = grid.send_and_receive([message])
    
    """The reply will contain the crypto context (cc), 
    that client's public key (pk) share,
    and the client's evaluation key (ek) share.
    """
    log(INFO, "Received reply.")
    reply = replies[0]
    cc_str = reply.content.config_records["key-dict"]['cc']
    pk_str = reply.content.config_records["key-dict"]['pk']
    ek_str = reply.content.config_records["key-dict"]['ek']

    # Server can save cc to file.
    # Convert to object, and then to file.
    # Should check sometime if you can go directly to file,
    # and whether that is faster.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    fhe_serialize_file(cc, storage+"cc", ST)

    """From here, server should use send_and_receive() to communicate
    with node_ids[1], ..., node_ids[n-2] 
    (assuming indexing from 0 to n-1 for number of clients n).
    Each client will receive cc, pk from previous client,
    sum of ek shares from previous clients and ek from client 0.
    I.e. the ek sum sent to client i is of the form
    (ek_0 + ek_1 + ... + ek_{i-1})
    In return, the server will get the clients pk and the ek sum
    (ek_0 + ek_1 + ... + ek_{i-1} + ek_i) from client i."""
    pk_list = [pk_str]
    ek_list = [ek_str]
    for i in range(1, n-1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                   'cc': cc_str,
                                   'pk': pk_list[i-1],
                                   'sum_ek': ek_list[i-1],
                                   'org_ek': ek_str})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_forward_keygen",  
            # target 'continue_forward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        # The server can send message and wait for reply.
        log(INFO, f"Sending message to node {i} to further generate "
            + "key pairs and evaluation key.")
        replies = grid.send_and_receive([message])
        log(INFO, "Received reply.")
        reply = replies[0]
        pk_list.append(reply.content.config_records["key-dict"]['pk'])
        ek_list.append(reply.content.config_records["key-dict"]['ek'])

    """The final client will receive the same parts
    as all the others, but the reply will be different.
    The last pk that the server receives is the joint public key (jpk).
    The server will also receive the sum of all ek shares (sum_ek),
    (ek_0 + ek_1 + ... + ek_{n-1}) and the ek product (prod_ek)
    ek_{n-1}*(ek_0 + ek_1 + ... + ek_{n-1}).
    """
    recordset = RecordDict()
    key_record = ConfigRecord({'id': n-1,
                               'cc': cc_str,
                               'pk': pk_list[-1],
                               'sum_ek': ek_list[-1],
                               'org_ek': ek_str})
    recordset['key-info'] = key_record
    message = Message(
        content=recordset,
        message_type="query.end_forward_keygen",  
        # target 'end_forward_keygen' method in ClientApp
        dst_node_id=node_ids[n-1],
        group_id="cont",
    )
    # The server can send message and wait for reply.
    log(INFO, f"Sending message to node {n-1} to generate "
        + "joint public key and continue generation of evaluation key.")
    replies = grid.send_and_receive([message])
    reply = replies[0]
    jpk_str = reply.content.config_records["key-dict"]['jpk']
    sum_ek_str = reply.content.config_records["key-dict"]['sum_ek']
    prod_ek_str = reply.content.config_records["key-dict"]['prod_ek']

    """Now the backwards part must be done
    to distribute the jpk and compute
    the joint evaluation key (jek).
    
    Each client (n-2, n-3, ..., 0)
    should receive the jpk, the sum_ek
    and the prod_ek from the previous client.
    The clients should reply with their new
    prod_ek.
    I.e., client i replies with
    (ek_i + ek_{i+1} + ... + ek_{n-1})*(ek_0 + ... + ek_{n-1}).
    
    The reply from client 0 is the jek."""
    sum_ek_list = [prod_ek_str]
    for i in range(n-2, -1, -1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                   'jpk': jpk_str,
                                   'org_sum_ek': sum_ek_str,
                                   'prev_sum_ek': sum_ek_list[n-2-i]})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_backward_keygen",  
            # target 'continue_backward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        # The server can send message and wait for reply.
        log(INFO, f"Sending message to node {i} to distribute jpk "
            + "and finish generating evaluation key.")
        replies = grid.send_and_receive([message])
        reply = replies[0]
        sum_ek_list.append(reply.content.config_records["key-dict"]['new_sum_ek'])

    # At this point, sum_ek_list[n-1] contains the jek.
    jek_str = sum_ek_list[-1]

    # Server saves jek to file.
    # Convert to object, then to file.
    log(INFO, "Serializing evaluation key.")
    jek = fhe_deserialize_string(jek_str, "EvalKey", ST)
    fhe_serialize_file(jek, storage+"jek", ST)

    log(INFO, "Key generation finished!")
    """Assuming that the clients do their computations
    before encrypting the results,
    only the server actually needs the evaluation key.
    Thus, it does not need to be distributed."""

def server_keygen_noise_flooding_noek(grid: Grid, context: Context, mode: str) -> None:
    """Version of server keygen which does not create evaluation key."""

    """OpenFHE parameters"""
    # Serialization type
    st = context.run_config['ser-type']
    if st == "BINARY":
        ST = fhe.BINARY
    else:
        ST = fhe.JSON
    # Scheme to use
    SCHEME = context.run_config['scheme']
    # Level for scheme.
    MULT_DEPTH = context.run_config['multiplicative-depth']
    """Should probably make ST, SCHEME and MULT_DEPTH
    choosable by user, both here and for master."""
    securityLevel = context.run_config['security-level']


    log(INFO, "")  # Add newline for log readability
    log(INFO, "Starting key generation.")

    """This setup assumes that there exists 
    a directory 'storage/key_store'
    with a subdirectory 'server'"""
    log(INFO, "Deleting old key files.")
    storage = os.path.abspath('storage/') + "/"
    if os.path.isfile(storage+"cc"):
        os.remove(storage+"cc")
    if(os.path.isfile(storage+"jek")):
        os.remove(storage+"jek")

    # Loop and wait until enough nodes are available.
    # Need every node available, since each must participate in keygen.
    min_nodes = context.run_config['num-clients']
    node_ids = []
    while len(node_ids) < min_nodes:
        node_ids = list(grid.get_node_ids())
        log(INFO, "Waiting for nodes to connect...")
        time.sleep(2)
    
    n = len(node_ids)

    log(INFO, "All nodes have connected.")

    """Pick an arbitrary client to start key generation.
    An obvious choice is the first client in node_ids.
    This client will get an initial message to start key generation."""
    recordset = RecordDict()
    cc_record = ConfigRecord({'id': 0, 'mode': mode})
    recordset['start-info'] = cc_record
    message = Message(
        content=recordset,
        message_type="query.start_forward_keygen",  
        # target 'start_forward_keygen' method in ClientApp
        dst_node_id=node_ids[0],
        group_id="start",
    )
    log(INFO, f"Sending message to node 0 to begin key generation.")
    log(INFO, f"Parameters are: \n" 
                f"\t\tscheme = {SCHEME}, \n"
                f"\t\tnum-clients = {min_nodes}, \n"
                f"\t\tserialzation type = {ST}, \n"
                f"\t\tmultdepth = {MULT_DEPTH}, \n"
                f"\t\tsecurityLevel = {securityLevel}.")
    # The server can send message and wait for reply.
    replies = grid.send_and_receive([message])
    
    """The reply will contain the crypto context (cc) and 
    that client's public key (pk) share..
    """
    log(INFO, "Received reply.")
    reply = replies[0]
    cc_str = reply.content.config_records["key-dict"]['cc']
    pk_str = reply.content.config_records["key-dict"]['pk']

    # Server can save cc to file.
    # Convert to object, and then to file.
    # Should check sometime if you can go directly to file,
    # and whether that is faster.
    cc = fhe_deserialize_string(cc_str, "CryptoContext", ST)
    fhe_serialize_file(cc, storage+"cc", ST)

    """From here, server should use send_and_receive() to communicate
    with node_ids[1], ..., node_ids[n-2] 
    (assuming indexing from 0 to n-1 for number of clients n).
    Each client will receive cc and pk share from previous client."""
    pk_list = [pk_str]
    for i in range(1, n-1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                   'cc': cc_str,
                                   'pk': pk_list[i-1]})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_forward_keygen",  
            # target 'continue_forward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        # The server can send message and wait for reply.
        log(INFO, f"Sending message to node {i} to further generate "
            + "key pairs and evaluation key.")
        replies = grid.send_and_receive([message])
        log(INFO, "Received reply.")
        reply = replies[0]
        pk_list.append(reply.content.config_records["key-dict"]['pk'])

    """The final client will receive the same parts
    as all the others, but the reply will be different.
    The last pk that the server receives is the joint public key (jpk).
    """
    recordset = RecordDict()
    key_record = ConfigRecord({'id': n-1,
                               'cc': cc_str,
                               'pk': pk_list[-1]})
    recordset['key-info'] = key_record
    message = Message(
        content=recordset,
        message_type="query.end_forward_keygen",  
        # target 'end_forward_keygen' method in ClientApp
        dst_node_id=node_ids[n-1],
        group_id="cont",
    )
    # The server can send message and wait for reply.
    log(INFO, f"Sending message to node {n-1} to generate "
        + "joint public key and continue generation of evaluation key.")
    replies = grid.send_and_receive([message])
    reply = replies[0]
    jpk_str = reply.content.config_records["key-dict"]['jpk']

    """Now the backwards part must be done
    to distribute the jpk.
    
    Each client (n-2, n-3, ..., 0)
    should receive the jpk."""
    messages = []
    for i in range(0, n-1):
        recordset = RecordDict()
        key_record = ConfigRecord({'id': i,
                                    'jpk': jpk_str})
        recordset['key-info'] = key_record
        message = Message(
            content=recordset,
            message_type="query.continue_backward_keygen",  
            # target 'continue_backward_keygen' method in ClientApp
            dst_node_id=node_ids[i],
            group_id="cont",
        )
        messages.append(message)
    # The server can send message and wait for reply.
    log(INFO, f"Sending message to node {i} to distribute jpk "
        + "and finish generating evaluation key.")
    grid.send_and_receive(messages)

    log(INFO, "Key generation finished!")
    """Assuming that the clients do their computations
    before encrypting the results,
    only the server actually needs the evaluation key.
    Thus, it does not need to be distributed."""
