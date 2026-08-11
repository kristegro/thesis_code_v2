import numpy as np

def encode_int(num, num_coeffs, base=3):
    """Encode an integer as a polynomial in the 
    cyclotomic ring Z[X]/(X^num_coeffs + 1) for a power of 2 num_coeffs,
    as outlined in thesis.
    
    Args:
        num (float): The real number to encode.

        num_coeffs (int): The number of coefficients that can be used for encoding,
            i.e., the number of elements in the list storing the coeffs. 
            All polynomials have a degree 0 <= deg <= num_coeffs-1.

            The polynomials can encode any integer L for which 
            ceil(log(2*L + 1) / log(base)) - 1
            is at most num_coeffs-1.
        
        base (int): The base to encode to. All coefficients in the polynomial
            are in the interval [-(base-1)/2, (base-1)/2]. Defaults to 3.
    
    Returns:
        list[int]: A list of coefficients for the encoding polynomial.
    """

    pol = np.zeros(num_coeffs, dtype=np.int32)
    # print(f"pol = {pol}")

    """For the integer part, convert to base by dividing repetedly and noting
    remainders. Last remainder is first digit in base representation and so on.
    
    E.g. 6_{10} = 20_{3} since
    6 = 2*3 + 0
    2 = 0*3 + 2"""
    pos = num_coeffs - 1
    while np.abs(num) > 0:
        # print(f"integer = {integer}, r = {integer%base}, new = {integer//base}")
        # print(f"num = {num}")
        # print(f"num%base = {num%base}")
        pol[pos] = num%base # Modulo to find remainder.
        """Makeshift løsning som ser ut til å fungere.
        Må jeg ha noe lignende for -2? Kan man i det hele tatt
        få num%3 = -2?
        Fungerer det å bare sjekke om num < 0?"""
        if num < 0:
            pol[pos] = num%(-base)
        num = int(num / base) # Intger division to remove remainder.
        """Tydeligvis fungerer ikke integer division // på negative tall,
        eks -1//3 = -1 istedenfor 0, mens 1//3 = 0 som forventet."""
        # print(f"num//base = {num}")
        pos -= 1
        # print(f"pol = {pol}")

    """Lastly, the polynomial must be convert from
    unbalanced base encoding to balanced,
    i.e., coefficients must be in the interval [-(base-1)/2, (base-1)/2].
    For base 3, simply convert 2 into 1T, where T = -1,
    by inserting T into the position of 2 and incrementing the left position by one.
    E.g., 0.22 = 1.T + 0.1T = 1.0T.
    
    Similarly, for base 3, convert 3 into 10 by insterting 0 into the position
    of 3 and incrementing the left position by one.
    E.g., 0.3 = 1.0"""
    # print(f"before conversion {pol}")

    # Integer part
    for i in range(num_coeffs-1, 0, -1): # n-1, n-2, ..., 1.
        if pol[i] > 1:
            pol[i]  = pol[i] - 3
            pol[i-1] += 1
        if pol[i] < -1:
            pol[i]  = pol[i] + 3
            pol[i-1] -= 1
    
    # print(f"after conversion {pol}")

    # I do not cover the case of pol[0], i.e.,
    # the left-most coefficient of the integer polynomial, being 2 or 3,
    # since if max_deg is large enough this should never occur.

    # Convert pol to list for compability with OpenFHE.
    pol = pol.tolist()
    # print(f"pol = {pol}")
    return pol

def decode_int(pol, base=3):
    """Decode the polynomial pol into the integer it represents.
    This is done according to the explanation in the thesis.
    
    Args:
        pol (list[int]): List of polynomial coefficients.
        
        base (int): Base used in encoding.
        
    Returns:
        float: The recovered real number."""
    
    # print(f"pol = {pol}")
    # print(f"p0  = {p0}")

    """The coefficients in pol are stored from highest to smallest,
    i.e., as pol_0*x^{n-1} + pol_1*x^{n-2} + ... + pol_{n-1}*x^0"""
    number = 0
    for i in range(1, len(pol)+1):
        # number = number + b_i * base^i
        number += pol[-i] * (base**(i-1))

    return int(number)

def encode_real(num, num_coeffs, max_deg_int, base=3):
    """Encode a real number as a polynomial in the 
    cyclotomic ring Z[X]/(X^num_coeffs + 1) for a power of 2 num_coeffs,
    as outlined in thesis.
    
    Args:
        num (float): The real number to encode.

        num_coeffs (int): The number of coefficients that can be used for encoding,
            i.e., the number of elements in the list storing the coeffs. 
            All polynomials have a degree 0 <= deg <= num_coeffs-1.

        max_deg_int (int): The upper bound on the degree of the integer polynomial,
            computed by ceil(log(2*L + 1) / log(3)) - 1
            for B=3, where L is upper bound on integer input.
        
        base (int): The base to encode to. All coefficients in the polynomial
            are in the interval [-(base-1)/2, (base-1)/2]. Defaults to 3.
    
    Returns:
        list[int]: A list of coefficients for the encoding polynomial.
    """

    pol = np.zeros(num_coeffs, dtype=np.int32)

    # Integer and fraction parts are managed separetly.
    integer = int(num)
    fraction = num - integer

    """For the integer part, convert to base by dividing repetedly and noting
    remainders. Last remainder is first digit in base representation and so on.
    
    E.g. 6_{10} = 20_{3} since
    6 = 2*3 + 0
    2 = 0*3 + 2"""
    pos = num_coeffs - 1
    # print(pos)
    while np.abs(integer) > 0:
        # print(f"integer = {integer}, r = {integer%base}, new = {integer//base}")
        # print(f"num = {integer}")
        # print(f"num%base = {integer%base}")
        pol[pos] = integer%base # Modulo to find remainder.
        """Makeshift løsning som ser ut til å fungere.
        Må jeg ha noe lignende for -2? Kan man i det hele tatt
        få num%3 = -2?
        Fungerer det å bare sjekke om num < 0?"""
        if integer < 0:
            pol[pos] = integer%(-base)
        integer = int(integer / base) # Intger division to remove remainder.
        """Tydeligvis fungerer ikke integer division // på negative tall,
        eks -1//3 = -1 istedenfor 0, mens 1//3 = 0 som forventet."""
        # print(f"num//base = {integer}")
        pos -= 1
        # print(f"pol = {pol}")
        # print(f"abs num = {np.abs(integer)}")
    
    """For the fractional part, convert to base by 
    multypling the fraction base, noting the integer part of the result,
    then multiply the fractional part of the result again.
    Continue this until the fraction becomes (approximately) 0,
    or until fc slots have been filled. First recorded integer part is the 
    first coefficient and so on.
    
    As an example, consider 0.37 with base 3. This gives
    0.37 * 3 = 1.11
    0.11 * 3 = 0.33
    0.33 * 3 = 1.00 (approximately)
    
    which gives 0.37_{10} = 0.101_{3}."""
    pos = 0 
    """Stop when you hit the start of the integer polynomial,
    which occupies the last max_dex + 1 positions."""
    # while np.abs(fraction) > 0.01 and pos < n - max_deg - 1:
    while pos < num_coeffs - max_deg_int - 1:
        whole = fraction*base
        # Will this if test impact performance when code is called thousands of times?
        """Jo flere 9ere jo større tall som er nærme 1 kan representeres."""
        # if (whole - int(whole)) >= 0.9999999:
        #     whole = int(whole) + 1 # If whole is close to nearest integer, round up.
        int_part = int(whole)
        pol[pos] = int_part
        # print(f"fraction = {fraction}, whole = {whole}")
        fraction = whole - int_part
        # print(f"whole = {whole}, int_part = {int_part}, fraction = {fraction}")
        pos += 1

    """Lastly, the polynomial must be convert from
    unbalanced base encoding to balanced,
    i.e., coefficients must be in the interval [-(base-1)/2, (base-1)/2].
    For base 3, simply convert 2 into 1T, where T = -1,
    by inserting T into the position of 2 and incrementing the left position by one.
    E.g., 0.22 = 1.T + 0.1T = 1.0T.
    
    Similarly, for base 3, convert 3 into 10 by insterting 0 into the position
    of 3 and incrementing the left position by one.
    E.g., 0.3 = 1.0"""
    # print(f"before conversion {pol}")
    # Fractional part
    for i in range(num_coeffs-max_deg_int-2, -1, -1): # n-max_deg-2, n-max_deg-3, ..., 0.
        # Will the if-test impact perfomance?
        if pol[i] > 1: # If pol[i] is 2 or 3
            pol[i] = pol[i] - 3 # 2 is converted to -1 and 3 to 0.
            pol[i-1] += 1 # Next position is incremented by one.
        if pol[i] < -1:
            pol[i] = pol[i] + 3
            pol[i-1] -= 1

    # Integer part
    for i in range(num_coeffs-1, num_coeffs-max_deg_int-1, -1): # n-1, n-2, ..., n-max_deg.
        if pol[i] > 1:
            pol[i]  = pol[i] - 3
            pol[i-1] += 1
        if pol[i] < -1:
            pol[i]  = pol[i] + 3
            pol[i-1] -= 1
    
    # print(f"after conversion {pol}")

    # I do not cover the case of pol[n-max_deg-1], i.e.,
    # the left-most coefficient of the integer polynomial, being 2 or 3,
    # since if max_deg is large enough this should never occur.

    # Convert pol to list for compability with OpenFHE.
    pol = pol.tolist()
    return pol


def decode_real(pol, max_deg_int, base=3):
    """Decode the polynomial pol into the real number it represents.
    This is done according to the explanation in the thesis.
    
    Args:
        pol (list[int]): List of polynomial coefficients.

        max_deg_int (int): The upper bound on the degree of the integer polynomial,
            computed by ceil(log(2*L + 1) / log(3)) - 1
            for B=3, where L is upper bound on integer input.
        
        base (int): Base used in encoding.
        
    Returns:
        float: The recovered real number."""
    
    n = len(pol)
    # Split pol into p0 and p1.
    # p0 occupies the last max_deg_int + 1 indices.
    # Last index is not included, so n-2-max_deg_intg means stopping at n-1-max_deg_int
    # and n-max_deg_int at n-max_deg_int+1
    p0 = pol[n-max_deg_int-1::]
    p1 = pol[0:n-max_deg_int-1]

    # print(f"pol = {pol}")
    # print(f"p0  = {p0}")
    # print(f"p1  = {p1}")

    number = 0
    """The coefficients in p0 are stored from highest to smallest,
    i.e., as p0_0*x^{n-1} + p0_1*x^{n-2} + ... + p0_{n-1}*x^0"""
    for i in range(1, len(p0)+1):
        # number = number + b_i * base^i
        number += p0[-i] * (base**(i-1))

    for i in range(1, len(p1)+1):
        # number = number + b_{i-1} * base^{-i}
        number += p1[i-1] * (base**(-i))

    return number


    
if __name__ == "__main__":
    # p = encode_real(6.370, 8, 2)
    # print(f"p(x) = {p}")
    # print(f"Recovered number = {decode_real(p, 2)}")

    # p_mark = encode_real(3.63, 8, 2)
    # print(f"p'(x) = {p_mark}")
    # print(f"Recovered number = {decode_real(p_mark, 2)}")

    # """Encoding er ikke gjort riktig.
    # Det er fractional part som skal encodes 
    # i de øvre koeffisientene og integer part som 
    # skal encodes i de nedre, som er motsatt av hva jeg har gjort.
    # Se overleaf og kladdebok.
    # Eller, jeg vet ikke helt må teste til jeg er sikker på at alt fungerer."""

    # # import time

    # # start = time.time()
    # # for _ in range(4386178):
    # #     encode_real(3.63, 32, 11)
    # #     decode_real([0]*32, 11)
    # # print(f"Time to do 4 386 178 encodings and decodings is {time.time()-start}")
    # """Encoding and decoding 4 386 178 weights will add around 33 seconds to runtime for each round,
    # but since runtime for a single round is already 10+ minutes, this is not too bad.
    
    # 4 386 178 is the total number of weights in the current dataset."""

    # q = encode_int(10, 8, 2)
    # print(f"q = {q}")
    # print(f"Recovered number = {decode_int(q, 2)}")


    # """Ett problem var at tall som var nærme null ble runda ned. Dette er nå løst."""
    # p_small = encode_real(0.0003, 32, 0)
    # print(f"p_small = {p_small}")
    # print(f"Recovered number = {decode_real(p_small, 0)}")

    # # test_org = 0.0003
    # # test = test_org
    # # i = 1
    # # while test < 1.0:
    # #     test = test_org*(3**i)
    # #     print(f"{test_org}*3^{i} = {test}")
    # #     i += 1

    # # print(f"For test = {test_org}, need i = {i-1} iterations of multiplication for test to exceed 1.")

    # """Et annet problem er at tall som er nærme 1, men litt mindre, rundes opp."""
    # p_close1 = encode_real(0.9991, 32, 0)
    # print(f"p_close1 = {p_close1}")
    # print(f"Recovered number = {decode_real(p_close1, 0)}")

    """Nåværende problem er at heltallsdelen til negative tall ser ut til å forsvinne."""
    # print(-6.14)
    # encoding = encode_real(-6.14, 16, 2)
    # print(encoding)
    # recovered = decode_real(encoding, 2)
    # print(recovered)
    # print(-6)
    # print(encode_int(-6, 4, 2))
    # print(decode_int(encode_int(-6, 4, 2), 2))

    """Nåværende problem er at resultatet av 
    encode_int(12000) + encode_int(12000) + encode_int(12000)
    ikke kan encodes."""
    # pol = encode_real(12000.3333333333333333, 16, 12)
    # print(f"pol = {pol}")
    # # Lat som at addisjon gjøres for å få 36000.
    # for i in range(len(pol)):
    #     pol[i] *= 3
    # print(f"pol = {pol}")
    # print(decode_real(pol, 12))

    """Nåværende problem er at noen encodings inneholder -2."""
    num_coeffs = 16
    max_deg_int = 10
    real2 = -12123.213
    pol2 = encode_real(real2, num_coeffs, max_deg_int)
    print(pol2)
    decode2 = decode_real(pol2, max_deg_int)
    print(decode2)
    """Fiksa nå. Manglet if-test for om fraction encoding har elementer < -1."""