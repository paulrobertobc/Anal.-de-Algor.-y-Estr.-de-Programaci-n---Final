import subprocess
from dataclasses import dataclass, field
from heapq import heappush, heappop, heapify
from time import perf_counter
from typing import Optional

PRIORIDADES = {"emergencia": 1000, "urgente": 100, "leve": 10}
ESPECIALIDADES = ("cardiologia", "pediatria", "traumatologia", "medicina general", "obstetricia", "cirugia",)

def hora_a_minutos(hora: str) -> int: #recibe un string y devuele un entero
    try:
        h, m = map(int, hora.split(":")) #con split vamos a cortar ese string en dos partes. despues con el map le decimos que cada parte sera un int
    except (ValueError, AttributeError):
        raise ValueError("La hora debe tener formato HH:MM, por ejemplo 08:00.")
    if not (0 <= h <= 23 and 0 <= m <= 59): #evaluamos si h entra dentro del rango 24 (porquw son las horas) y m entra en el rango de 59 (porque son los minutos)
        raise ValueError("La hora debe estar entre 00:00 y 23:59.")
    return h * 60 + m #al final devuelve el calculo con los minutos finales

def minutos_a_hora(minutos: int) -> str: #recibe un int y devuelve un string
    minutos = minutos % (24 * 60) # dividimos los minutos entre 24 * 60 (que es 1440, los minutos totales de un dia) para mantenernos en el rango de un dia
    return f"{minutos // 60:02d}:{minutos % 60:02d}" #devolvemos las horas con la division entera. delvolvemos los minutos restantes usando el modulo (%)

@dataclass
class Turno:
    hora_inicio: str
    hora_fin: str

    def __post_init__(self):
        inicio = hora_a_minutos(self.hora_inicio)
        fin = hora_a_minutos(self.hora_fin)
        if fin == inicio:
            raise ValueError("El inicio y el fin del turno no pueden ser iguales.")

    @property
    def inicio(self):
        return hora_a_minutos(self.hora_inicio)

    @property
    def fin(self):
        return hora_a_minutos(self.hora_fin)

    @property
    def fin_relativo(self):
        """Fin del turno en la misma jornada; suma 24 h si cruza medianoche."""
        fin = self.fin
        return fin + 24 * 60 if fin < self.inicio else fin

@dataclass
class Medico:
    Id: int
    Nombre: str
    Especialidad: str
    Turno: Turno
    estado: str = "Fuera de turno"
    consultorio_id: Optional[int] = None
    ocupado_hasta: int = -1
    extension: bool = False
    fin_turno_asignado: int = -1

@dataclass
class Consultorio:
    Id: int
    ocupado_por: Optional[int] = None

@dataclass
class Paciente:
    Id: int
    Nombre: str
    Llegada: int  # minutos desde el inicio de la jornada simulada
    Duracion: int
    Prioridad: str
    Especialidad: str = "medicina general"
    NecesitaCama: bool = False
    NecesitaDiagnostico: bool = False
    inicio: int = -1
    fin: int = -1
    espera: int = 0
    estado: str = "Pendiente"
    medico_id: Optional[int] = None
    consultorio_id: Optional[int] = None

    @property
    def puntuacion(self):
        return PRIORIDADES[self.Prioridad]

def calcular_beneficio(p, ahora):
    espera_actual = max(0, ahora - p.Llegada)
    return p.puntuacion + min(espera_actual, 60)

def optimizar_programacion_dinamica(pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora):
    """DP exacta: cada estado reserva médicos concretos mediante una máscara de bits."""
    # (máscara médicos, camas, diagnósticos) -> (beneficio, tupla (índice paciente, índice médico))
    dp = {(0, 0, 0): (0, ())}
    for i, p in enumerate(pacientes):
        nuevos = dict(dp)  # alternativa: no seleccionar al paciente
        beneficio = calcular_beneficio(p, ahora)
        for (mascara, usadas_camas, usados_diag), (valor, asignaciones) in dp.items():
            if mascara.bit_count() >= consultorios:
                continue
            for indice_medico in medicos_elegibles.get(i, []):
                bit = 1 << indice_medico
                if mascara & bit:
                    continue
                nc = usadas_camas + int(p.NecesitaCama)
                nd = usados_diag + int(p.NecesitaDiagnostico)
                if nc > camas or nd > diagnosticos:
                    continue
                estado = (mascara | bit, nc, nd)
                candidato = (valor + beneficio, asignaciones + ((i, indice_medico),))
                if estado not in nuevos or candidato[0] > nuevos[estado][0]:
                    nuevos[estado] = candidato
        dp = nuevos
    mejor = max(dp.values(), key=lambda x: x[0])
    return list(mejor[1]), mejor[0]


def optimizar_divide_y_venceras(pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora):
    """Divide pacientes en dos mitades, genera fronteras de estados y combina máscaras compatibles."""
    def generar_estados(inicio, fin):
        estados = {(0, 0, 0): (0, ())}
        for i in range(inicio, fin):
            p = pacientes[i]
            nuevos = dict(estados)
            beneficio = calcular_beneficio(p, ahora)
            for (mascara, usadas_camas, usados_diag), (valor, asignaciones) in estados.items():
                if mascara.bit_count() >= consultorios:
                    continue
                for indice_medico in medicos_elegibles.get(i, []):
                    bit = 1 << indice_medico
                    if mascara & bit:
                        continue
                    nc = usadas_camas + int(p.NecesitaCama)
                    nd = usados_diag + int(p.NecesitaDiagnostico)
                    if nc > camas or nd > diagnosticos:
                        continue
                    estado = (mascara | bit, nc, nd)
                    candidato = (valor + beneficio, asignaciones + ((i, indice_medico),))
                    if estado not in nuevos or candidato[0] > nuevos[estado][0]:
                        nuevos[estado] = candidato
            estados = nuevos
        return estados

    mitad = len(pacientes) // 2
    izquierda = generar_estados(0, mitad)
    derecha = generar_estados(mitad, len(pacientes))
    mejor_beneficio = 0
    mejor_asignacion = ()
    for (mascara_i, camas_i, diag_i), (valor_i, asign_i) in izquierda.items():
        for (mascara_d, camas_d, diag_d), (valor_d, asign_d) in derecha.items():
            if mascara_i & mascara_d:
                continue
            if (mascara_i | mascara_d).bit_count() > consultorios:
                continue
            if camas_i + camas_d > camas or diag_i + diag_d > diagnosticos:
                continue
            total = valor_i + valor_d
            if total > mejor_beneficio:
                mejor_beneficio = total
                mejor_asignacion = asign_i + asign_d
    return list(mejor_asignacion), mejor_beneficio


def comparar_algoritmos(pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora):
    inicio = perf_counter()
    asign_pd, beneficio_pd = optimizar_programacion_dinamica(
        pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora)
    tiempo_pd = perf_counter() - inicio
    inicio = perf_counter()
    asign_dv, beneficio_dv = optimizar_divide_y_venceras(
        pacientes, medicos_elegibles, camas, diagnosticos, consultorios, ahora)
    tiempo_dv = perf_counter() - inicio

    def detalle(nombre, asignaciones, beneficio, duracion):
        elegidos = [pacientes[i] for i, _ in asignaciones]
        print(f"\n            {nombre}            ")
        print(f"- Beneficio total: {beneficio}")
        print(f"- Pacientes seleccionados: {len(elegidos)}")
        print(f"- Médicos utilizados: {len(asignaciones)}")
        print(f"- Camas utilizadas: {sum(p.NecesitaCama for p in elegidos)}/{camas}")
        print(f"- Equipos de diagnóstico utilizados: {sum(p.NecesitaDiagnostico for p in elegidos)}/{diagnosticos}")
        print(f"- Consultorios utilizados: {len(asignaciones)}/{consultorios}")
        print(f"- Tiempo de ejecución: {duracion:.8f} segundos")
        print("Selección:", ", ".join(p.Nombre for p in elegidos) or "Ninguno")

    print("\n" + "COMPARACIÓN DE ALGORITMOS".center(100, "=") + "\n")
    print(f"Pacientes candidatos: {len(pacientes)} | Minuto de simulación: {ahora}")
    detalle("PROGRAMACIÓN DINÁMICA", asign_pd, beneficio_pd, tiempo_pd)
    detalle("DIVIDE Y VENCERÁS", asign_dv, beneficio_dv, tiempo_dv)
    print("\n" + "CONCLUSIÓN DE LA COMPARACIÓN".center(100, "="))
    if beneficio_pd == beneficio_dv:
        print("| Ambos algoritmos encontraron el mismo beneficio óptimo.")
    else:
        print("| Los resultados difieren; revisa las selecciones y restricciones.")
    if tiempo_pd < tiempo_dv:
        print("| En esta prueba, Programación Dinámica fue más rápida.")
    elif tiempo_dv < tiempo_pd:
        print("| En esta prueba, Divide y Vencerás fue más rápido.")
    else:
        print("Ambos tardaron prácticamente lo mismo.")
    print("=" * 100)
    return asign_pd, asign_dv


class Hospital:
    def __init__(self, medicos=0, camas=5, diagnosticos=3, cantidad_consultorios=4):
        self.total_camas = camas
        self.total_diagnosticos = diagnosticos
        self.consultorios = [Consultorio(i + 1) for i in range(cantidad_consultorios)]
        self.medicos = []
        self.pacientes = []
        self.siguiente_id = 1
        self.siguiente_medico_id = 1
        self.tiempo = 0
        self.inicio_jornada = None  # se fija con el primer médico registrado.
        self.ejecutado = False
        datosMedicos = [# ("cardiologia", "pediatria", "traumatologia", "medicina general", "obstetricia", "cirugia",)
            ("Victor Caballero", "cardiologia", "08:00", "20:00"),#turno mañana
            ("Mayra Torres", "pediatria", "08:00", "20:00"),
            ("John Vargas", "medicina general", "08:00", "20:00"),
            ("Gbariel Salazar", "cirugia", "08:30", "20:30"),
            ("Pedro Mendoza", "traumatologia", "08:40", "20:50"),
            ("Paula Jaramillo", "pediatria", "20:00", "08:00"),#turno noche
            ("Lucia Lozano", "obstetricia", "20:30", "08:30"),
            ("Leandro Paredes", "cardiologia", "20:00", "08:00"),
            ("Victoria Rojas", "traumatologia", "20:00", "08:00"),
            ("Mercedes Cabrera", "medicina general", "20:00", "08:00"),
            ("Paul Cardenas", "cirugia", "20:30", "08:30"),
            ("Ester Campos", "obstetricia", "20:30", "08:30")
        ]
        for m in datosMedicos:
            self.registrar_medico(*m)

    @property
    def total_medicos(self):
        return len(self.medicos)

    def registrar_medico(self, nombre, especialidad, hora_inicio, hora_fin):
        especialidad = especialidad.strip().lower()
        if especialidad not in ESPECIALIDADES:
            raise ValueError("Especialidad no válida.")
        turno = Turno(hora_inicio, hora_fin)
        medico = Medico(self.siguiente_medico_id, nombre.strip(), especialidad, turno)
        self.medicos.append(medico)
        self.siguiente_medico_id += 1
        if self.inicio_jornada is None:
            self.inicio_jornada = turno.inicio
        else:
            self.inicio_jornada = min(self.inicio_jornada, turno.inicio)
        return medico

    def registrar_paciente(self, nombre, llegada, duracion, prioridad, especialidad="medicina general",necesita_cama=False, necesita_diagnostico=False):
        if llegada < 0 or duracion <= 0:
            raise ValueError("La llegada no puede ser negativa y la duración debe ser mayor que cero.")
        if prioridad not in PRIORIDADES:
            raise ValueError("Prioridad no válida.")
        especialidad = especialidad.strip().lower()
        if especialidad not in ESPECIALIDADES:
            raise ValueError("Especialidad no válida.")
        if necesita_cama and self.total_camas == 0:
            raise ValueError("El hospital no dispone de camas.")
        if necesita_diagnostico and self.total_diagnosticos == 0:
            raise ValueError("No hay equipos de diagnóstico disponibles.") 
        p = Paciente(self.siguiente_id, nombre.strip(), llegada, duracion, prioridad,especialidad, necesita_cama, necesita_diagnostico)
        self.pacientes.append(p)
        self.siguiente_id += 1
        return p

    def hora_actual(self, tiempo=None):
        if tiempo is None:
            tiempo = self.tiempo
        if self.inicio_jornada is None:
            return "00:00"
        return minutos_a_hora(self.inicio_jornada + tiempo)

    def convertir_hora_llegada(self, hora: str) -> int:
        """Convierte HH:MM a minutos relativos al inicio de la simulación."""
        if self.inicio_jornada is None:
            raise ValueError("Primero debes registrar al menos un médico.")
        minutos_reloj = hora_a_minutos(hora)
        llegada = minutos_reloj - self.inicio_jornada
        if llegada < 0:
            llegada += 24 * 60
        return llegada

    def intervalo_turno_actual(self, medico, tiempo):
        #nos devuelve un tupla (inicio, fin) del turno que cubre el minuto simulado actual.
        #los valores devueltos son minutos relativos al inicio de la simulación, también contempla turnos que comenzaron el día anterior y terminan hoy
        absoluto = self.inicio_jornada + tiempo
        dia = absoluto // (24 * 60)
        inicio_dia = medico.Turno.inicio
        fin_dia = medico.Turno.fin_relativo
        for d in (dia - 1, dia, dia + 1):
            inicio_abs = d * 24 * 60 + inicio_dia
            fin_abs = d * 24 * 60 + fin_dia
            if inicio_abs <= absoluto < fin_abs:
                return (inicio_abs - self.inicio_jornada, fin_abs - self.inicio_jornada)
        return None

    def proximo_inicio_turno(self, medico, tiempo): #proximo inicio diario del turno, expresado en tiempo de simulacion
        absoluto = self.inicio_jornada + tiempo
        dia = absoluto // (24 * 60)
        inicio_abs = dia * 24 * 60 + medico.Turno.inicio
        if inicio_abs <= absoluto:
            inicio_abs += 24 * 60
        return inicio_abs - self.inicio_jornada

    def medicos_activos(self, especialidad=None):
        return [m for m in self.medicos if especialidad is None or m.Especialidad == especialidad]

    def actualizar_estados_medicos(self):
        for m in self.medicos:
            if m.estado == "Atendiendo":
                continue
            m.estado = "Disponible" if self.intervalo_turno_actual(m, self.tiempo) else "Fuera de turno"

    def consultorio_libre(self):
        return next((c for c in self.consultorios if c.ocupado_por is None), None)

    def medicos_elegibles(self, candidatos): #devuelve los indices de médicos que pueden iniciar cada paciente en este instante
        disponibles = []
        intervalos_fin = {}
        for m in self.medicos:
            intervalo = self.intervalo_turno_actual(m, self.tiempo)
            if m.estado == "Atendiendo" or intervalo is None:
                continue
            disponibles.append(m)
            intervalos_fin[m.Id] = intervalo[1]
        mapa = {}
        for i, p in enumerate(candidatos):
            opciones = []
            for indice, m in enumerate(disponibles):
                if m.Especialidad != p.Especialidad:
                    continue
                # solo una emergencia puede terminar despurs de la hora de salida
                if self.tiempo + p.Duracion <= intervalos_fin[m.Id] or p.Prioridad == "emergencia":
                    opciones.append(indice)
            mapa[i] = opciones
        return disponibles, mapa

    def recursos_libres(self, activos):
        camas_ocupadas = sum(p.NecesitaCama for _, p, _, _ in activos)
        diag_ocupados = sum(p.NecesitaDiagnostico for _, p, _, _ in activos)
        consultorios_libres = sum(c.ocupado_por is None for c in self.consultorios)
        return self.total_camas - camas_ocupadas, self.total_diagnosticos - diag_ocupados, consultorios_libres

    def puede_llegar_medico_especialidad(self, especialidad, ahora, activos): #comprueba si hay un medico que ya atiende o cuyo turno aun no termina a iniciar
        for m in self.medicos:
            if m.Especialidad != especialidad:
                continue
            if m.estado == "Atendiendo":
                return True
            if self.intervalo_turno_actual(m, ahora) is not None:
                return True
            # si su siguiente turno comienza pronto el paciente puede esperar
            if self.proximo_inicio_turno(m, ahora) - ahora <= 60:
                return True
        return False

    def trasladar_sin_cobertura(self, espera, activos):
        restantes = []
        for p in espera:
            if not self.puede_llegar_medico_especialidad(p.Especialidad, self.tiempo, activos):
                p.estado = "Trasladado"
                p.fin = self.tiempo
                print(f"[{self.hora_actual()}] TRASLADO: {p.Nombre} requiere {p.Especialidad}; no hay especialista disponible. Se deriva a otro hospital.")
            else:
                restantes.append(p)
        return restantes

    def asignar_consultorio(self, medico, paciente):
        consultorio = self.consultorio_libre()
        if consultorio is None:
            return False
        consultorio.ocupado_por = medico.Id
        medico.consultorio_id = consultorio.Id
        medico.estado = "Atendiendo"
        paciente.medico_id = medico.Id
        paciente.consultorio_id = consultorio.Id
        return True

    def simular(self, escoger_algoritmo):
        if self.ejecutado:
            print("La simulación ya fue procesada. Reinicie antes de volver a ejecutarla.")
            return
        if not self.medicos:
            print("Registra al menos un médico antes de iniciar la simulación.")
            return
        self.ejecutado = True
        llegadas = [] #crea una lista vacia para almacenar las llegadas de los pacientes

        for p in self.pacientes: #evaluamos cada paciente registrado en el hospital
            heappush(llegadas, (p.Llegada, p.Id, p)) #con heappush insertamos en la lista llegadas una tupla con la llegada del paciente, su id y el objeto paciente
        
        espera = [] #crea una lista vacia para almacenar los pacientes que estan en espera
        activos = []  # crea una lista vacia para almacenar los pacientes que estan siendo atendidos
        self.tiempo = 0 #inicializa el tiempo de simulacion en 0, que representa el inicio de la jornada hospitalaria
        
        print("\n" + "INICIO DE LA SIMULACIÓN".center(100, "-") + "\n")

        while llegadas or espera or activos: #iniciamos un bucle que se ejecuta mientras haya llegadas, pacientes en espera o pacientes activos
            # primero finalizar atenciones cuyo tiempo ya termino
            
            terminados = [a for a in activos if a[0] <= self.tiempo] #aquí filtramos los pacientes activos que han terminado su atención, 
            #es decir, aquellos cuyo tiempo de finalización es menor o igual al tiempo actual de simulación
            
            for fin, p, medico, consultorio in terminados: #evaluamos cada paciente que ha terminado su atención y obtenemos su tiempo de finalización, el 
                #objeto paciente, el médico que lo atendió y el consultorio utilizado
                p.fin = fin 
                p.estado = "Atendido"
                medico.estado = "Disponible" if self.intervalo_turno_actual(medico, self.tiempo) else "Fuera de turno"
                medico.consultorio_id = None
                medico.ocupado_hasta = -1
                extra = " (turno extendido por emergencia)" if medico.extension else ""
                medico.extension = False
                medico.fin_turno_asignado = -1
                consultorio.ocupado_por = None 
                print(f"[{self.hora_actual(fin)}] Finaliza {p.Nombre} con Dr./Dra. {medico.Nombre}, consultorio {consultorio.Id}{extra}.")
            
            activos = [a for a in activos if a[0] > self.tiempo] #actualizamos la lista de pacientes activos eliminando aquellos que han terminado su atención

            # segundo, registrar todas las llegadas del instante
            while llegadas and llegadas[0][0] <= self.tiempo: #mientras haya llegadas pendientes y la llegada del primer paciente en la lista sea menor o igual 
                #al tiempo actual de simulación
                _, _, p = heappop(llegadas) #con heappop extraemos la llegada del primer paciente de la lista y obtenemos el objeto paciente
                p.estado = "En espera"
                
                heappush(espera, (-p.puntuacion, p.Llegada, p.Id, p)) #con heappush insertamos en la lista de espera una tupla con la puntuación 
                #negativa del paciente, su llegada, su id y el objeto paciente
                
                print(f"[{self.hora_actual()}] Llega {p.Nombre}: {p.Prioridad.upper()} | requiere {p.Especialidad}.")

            self.actualizar_estados_medicos() #actualizamos el estado de los médicos disponibles según el tiempo actual de simulación
            
            espera = [(-p.puntuacion, p.Llegada, p.Id, p) for _, _, _, p in espera] #actualizamos la lista de espera con las puntuaciones negativas, 
            #llegadas, ids y objetos pacientes
            
            heapify(espera) #con heapify convertimos la lista de espera en una estructura de montículo (heap) para mantener el orden de prioridad
            candidatos = [item[3] for item in espera] #creamos una lista de candidatos a ser atendidos, extrayendo los objetos pacientes de la lista de espera
            
            disponibles, mapa_elegibles = self.medicos_elegibles(candidatos) #obtenemos la lista de médicos disponibles y un mapa de 
            #médicos elegibles para cada paciente candidato
            
            camas_libres, diagnosticos_libres, consultorios_libres = self.recursos_libres(activos) #obtenemos la cantidad de camas libres, equipos de diagnóstico
            #libres y consultorios libres según los pacientes activos para el tiempo actual de simulación

            if candidatos and disponibles and consultorios_libres > 0: #evaluamos si hay candidatos, médicos disponibles y consultorios libres para asignar pacientes
                if escoger_algoritmo == 1: #evaluamos el algoritmo seleccionado por el usuario, si es 1 se utiliza programación dinámica, si es 2 se utiliza divide y vencerás
                    #y se optimiza la asignación de pacientes a médicos y consultorios
                    asignaciones, _ = optimizar_programacion_dinamica(
                        candidatos, mapa_elegibles, camas_libres, diagnosticos_libres,
                        consultorios_libres, self.tiempo)
                else:
                    asignaciones, _ = optimizar_divide_y_venceras(
                        candidatos, mapa_elegibles, camas_libres, diagnosticos_libres,
                        consultorios_libres, self.tiempo)

                if asignaciones: #si hay asignaciones válidas, se procede a asignar pacientes a médicos y consultorios
                    
                    elegidos = {candidatos[i].Id for i, _ in asignaciones} #creamos un conjunto de ids de pacientes elegidos para ser atendidos
                    
                    espera = [item for item in espera if item[3].Id not in elegidos] #actualizamos la lista de espera eliminando a los pacientes que han sido asignados
                    
                    heapify(espera) #con heapify convertimos nuevamente la lista de espera en una estructura de montículo (heap) para mantener el orden de prioridad
                    
                    for indice_paciente, indice_medico in asignaciones: #evaluamos cada asignación de paciente a médico y obtenemos los índices correspondientes
                        p = candidatos[indice_paciente]
                        
                        medico = disponibles[indice_medico]
                        consultorio = self.consultorio_libre() #obtenemos un consultorio libre para asignar al paciente y al médico
                        
                        if consultorio is None: #si no hay consultorio libre, se vuelve a insertar el paciente en la lista de espera con una puntuación 
                            #negativa para mantener el orden de prioridad
                            # protección adicional, normalmente no ocurre por la restriccion del algoritmo
                            heappush(espera, (-p.puntuacion, p.Llegada, p.Id, p)) #con heappush insertamos nuevamente al paciente en la lista de espera
                            continue #le damos continue para volver al inicio del bucle y procesar la siguiente asignación
                        
                        self.asignar_consultorio(medico, p) #asignamos el consultorio al médico y al paciente, actualizando sus estados y atributos correspondientes
                        p.inicio = self.tiempo #actualizamos el tiempo de inicio de atención del paciente con el tiempo actual de simulación
                        p.fin = self.tiempo + p.Duracion #actualizamos el tiempo de finalización de atención del paciente sumando la duración de la atención al tiempo 
                        #actual de simulación
                        p.espera = p.inicio - p.Llegada
                        p.estado = "En atención" 
                        medico.ocupado_hasta = p.fin
                        
                        intervalo = self.intervalo_turno_actual(medico, self.tiempo) #obtenemos el intervalo de turno actual del médico para determinar si su 
                        #turno se extiende más allá del tiempo de finalización de atención del paciente
                        
                        fin_turno = intervalo[1] if intervalo else self.tiempo #si no hay intervalo de turno actual, se considera que el turno del médico 
                        #termina en el tiempo actual de simulación
                        
                        medico.fin_turno_asignado = fin_turno
                        medico.extension = p.fin > fin_turno
                        
                        if medico.extension: #si el tiempo de finalización de atención del paciente es mayor que el tiempo de finalización del
                            # turno del médico, se imprime un aviso indicando que el médico extenderá su turno para atender la emergencia
                            print(f"[{self.hora_actual()}] AVISO: Dr./Dra. {medico.Nombre} extenderá su turno para terminar esta emergencia.") #imprimosm
                        
                        activos.append((p.fin, p, medico, consultorio)) #agregamos el paciente, el médico y el consultorio a la lista de pacientes activos, indicando 
                        #el tiempo de finalización de atención del paciente
                        
                        print(f"[{self.hora_actual()}] Inicia {p.Nombre} con Dr. (a) {medico.Nombre} ({p.Especialidad.title()}) en consultorio {consultorio.Id}; termina {self.hora_actual(p.fin)}.")
                    
                    continue #con continue volvemos al inicio del bucle para procesar el siguiente evento de simulación

            espera_pacientes = [item[3] for item in espera] #si no hay asignaciones válidas, se evalúa la lista de pacientes en espera y se 
            #verifica si alguno de ellos no puede ser atendido por falta de cobertura médica
            
            trasladados_o_espera = self.trasladar_sin_cobertura(espera_pacientes, activos) #se llama a la función trasladar_sin_cobertura para determinar qué 
            #pacientes deben ser trasladados a otro hospital por falta de cobertura médica
            
            ids_restantes = {p.Id for p in trasladados_o_espera} #se crea un conjunto de ids de pacientes que aún pueden esperar a ser atendidos
            
            espera = [item for item in espera if item[3].Id in ids_restantes] #se actualiza la lista de espera eliminando a los pacientes que han sido 
            #trasladados a otro hospital
            
            heapify(espera) #con heapify convertimos nuevamente la lista de espera en una estructura de montículo (heap) para mantener el orden de prioridad
            # tercero, avanzar al proximo evento, que es llegada, fin de atención o inicio de turno
            
            proximos = [] #creamos una lista vacía para almacenar los tiempos de los próximos eventos que pueden ocurrir en la simulación
            
            if llegadas: #si hay pacientes pendientes de llegada, se agrega el tiempo de llegada del primer paciente en la lista de llegadas a la lista de próximos eventos
                proximos.append(llegadas[0][0])
            
            if activos: #si hay pacientes activos, se agrega el tiempo de finalización de atención del primer paciente en la lista de activos a la lista de próximos eventos
                proximos.append(min(a[0] for a in activos))
            
            for m in self.medicos: #se evalúa cada médico registrado en el hospital para determinar si hay algún evento relacionado con su turno que pueda 
                #ocurrir en la simulación
                if m.estado == "Atendiendo": #si el médico está atendiendo a un paciente, se omite la evaluación de su turno y se pasa al siguiente médico
                    continue
                
                inicio_siguiente = self.proximo_inicio_turno(m, self.tiempo) #se obtiene el tiempo de inicio del siguiente turno del médico, expresado en
                #minutos relativos al inicio de la simulación
                
                if inicio_siguiente > self.tiempo: #si el tiempo de inicio del siguiente turno del médico es mayor que el tiempo actual de simulación, se agrega 
                    #a la lista de próximos eventos
                    proximos.append(inicio_siguiente) #agregamos el tiempo de inicio del siguiente turno del médico a la lista de próximos eventos
                
                intervalo = self.intervalo_turno_actual(m, self.tiempo) #se obtiene el intervalo de turno actual del médico para determinar si su turno se extiende más 
                #allá del tiempo actual de simulación
                
                if intervalo and intervalo[1] > self.tiempo: #si el médico tiene un turno activo y su tiempo de finalización es mayor que el tiempo actual de 
                    #simulación, se agrega a la lista de próximos eventos
                    
                    proximos.append(intervalo[1]) #agregamos el tiempo de finalización del turno actual del médico a la lista de próximos eventos
            
            proximos = [t for t in proximos if t > self.tiempo] #se filtra la lista de próximos eventos para eliminar aquellos que ya han ocurrido o que son 
            #menores o iguales al tiempo actual de simulación
            
            if not proximos: #si la lista de próximos eventos está vacía, significa que no hay más eventos que puedan ocurrir en la simulación, por lo que se finaliza la
                #simulación

                # no quedan eventos que puedan liberar recursos o incorporar medicos.
                for _, _, _, p in espera: #por cada paciente que aún está en espera, se actualiza su estado a "Sin atención" y se imprime un mensaje indicando que 
                    #no fue posible atenderlo, con _ como variable de descarte para los valores que no se utilizan
                    
                    p.estado = "Sin atención" #actualizamos el estado del paciente a "Sin atención" para indicar que no fue posible atenderlo en la simulación
                    print(f"[{self.hora_actual()}] No fue posible atender a {p.Nombre}; queda sin atención en esta simulación.")
                
                espera.clear() #limpiamos la lista de espera para indicar que no hay más pacientes pendientes de atención
                break
            self.tiempo = min(proximos)

        print("\n" + "FIN DE LA SIMULACIÓN".center(100, "-") + "\n")
        self.mostrar_reporte()

    def reiniciar(self):
        self.tiempo = 0
        self.ejecutado = False
        for p in self.pacientes:
            p.inicio = -1
            p.fin = -1
            p.espera = 0
            p.estado = "Pendiente"
            p.medico_id = None
            p.consultorio_id = None
        for m in self.medicos:
            m.estado = "Fuera de turno"
            m.consultorio_id = None
            m.ocupado_hasta = -1
            m.extension = False
            m.fin_turno_asignado = -1
        for c in self.consultorios:
            c.ocupado_por = None
        print("El sistema hospitalario ha sido reiniciado.")

    def mostrar_medicos(self):
        print("\nPLANTILLA MÉDICA")
        for m in self.medicos:
            etiqueta = " (cruza medianoche)" if m.Turno.fin < m.Turno.inicio else ""
            print(f"#{m.Id} Dr./Dra. {m.Nombre:18} | {m.Especialidad.title():20} | {m.Turno.hora_inicio}-{m.Turno.hora_fin}{etiqueta}")

    def mostrar_reporte(self):
        atendidos = [p for p in self.pacientes if p.estado == "Atendido"]
        trasladados = [p for p in self.pacientes if p.estado == "Trasladado"]
        pendientes = [p for p in self.pacientes if p.estado not in ("Atendido", "Trasladado")]
        print("REPORTE ESTADÍSTICO: ")
        print(f"- Pacientes registrados: {len(self.pacientes)}")
        print(f"- Pacientes atendidos:   {len(atendidos)}")
        print(f"- Pacientes trasladados: {len(trasladados)}")
        print(f"- Pacientes pendientes:  {len(pendientes)}")
        if atendidos:
            esperas = [p.espera for p in atendidos]
            print(f"- Espera promedio:       {sum(esperas) / len(esperas):.2f} min")
            print(f"- Espera mínima:         {min(esperas)} min")
            print(f"- Espera máxima:         {max(esperas)} min")
        print("\nDETALLE POR PACIENTE: ")
        for p in sorted(self.pacientes, key=lambda x: x.Id):
            medico = next((m.Nombre for m in self.medicos if m.Id == p.medico_id), "-")
            consultorio = p.consultorio_id if p.consultorio_id is not None else "-"
            print(f"#{p.Id} {p.Nombre:18} | {p.Especialidad:20} | {p.estado:12} | Médico: {medico:15} | Consultorio: {consultorio} | Espera: {p.espera if p.inicio >= 0 else '-'}")


def datos_demostracion(hospital): #pacientes ejemplo para dar una demostracion
    datos = [ 
        #los numeros enteros que representan la llegada son los minutos desde que empezo la jornada
        #la jornada comienza a las 6,
        #0	 =>  06:00 a. m.
        #30	 => 06:30 a. m.
        #60	 => 07:00 a. m.
        #120	 => 08:00 a. m.
        #480	 => 02:00 p. m.
        #960	 => 10:00 p. m.
        #1080 => 12:00 a. m. (medianoche)
        #formato de objeto tipo Paciente: nombre (string), llegada(entero que representa los minutos desde el inicio de jornada. guiarse de la guia de arriba), 
        # duracion(minutos en entero), prioridad (string), necesitaCama (booleano), necesitaDiagnostico(booleano).
        #especialidades: #("cardiologia", "pediatria", "traumatologia", "medicina general", "obstetricia", "cirugia",)
        ("Paul", 5, 45, "emergencia", "cardiologia", True, False),
        ("Efrain Aguilar", 15, 90, "leve", "pediatria", False, False),
        ("Sebastian Lizarzaburu", 25, 80, "urgente", "cirugia", False, True),
        ("Cris Soifer", 35, 35, "urgente", "traumatologia", False, True),
        ("Jhonny Pacheco", 60, 120, "emergencia", "medicina general", False, True),
        ("Carlos Alcantara", 95, 60, "leve", "cardiologia", True, False),
        ("Isabella Merced", 105, 45, "leve", "obstetricia", False, False),
        ("Nicola Porcella", 130, 40, "urgente", "pediatria", False, False),
        ("Adrian Castillo", 140, 90, "urgente", "traumatologia", False, True),
        ("Mariana López", 155, 90, "urgente", "cardiologia", False, True),
        ("Mateo", 180, 150, "emergencia", "cardiologia", True, False),
        ("Ricardo Mendoza", 180, 120, "emergencia", "medicina general", True, True),
        ("Nicolas Flores", 195, 130, "emergencia", "cirugia", True, True),
        ("Valentina Cruz", 200, 45, "leve", "obstetricia", False, False),
        ("Lucia", 225, 35, "urgente", "medicina general", False, True),
        ("Diego Herrera", 225, 80, "urgente", "traumatologia", False, True),
        ("Camila Fernández", 240, 55, "leve", "obstetricia", False, False),
        ("Rodrigo", 270, 75, "leve", "traumatologia", False, False),
        ("Andrés Castillo", 270, 150, "emergencia", "cirugia", True, True),
        ("Paola Jimenez", 285, 100, "urgente", "obstetricia", False, True),
        ("Lucía Ramírez", 290, 40, "leve", "cardiologia", False, False),
        ("Diego Ramirez", 305, 150, "emergencia", "traumatologia", True, True),
        ("Valeria", 315, 50, "emergencia", "pediatria", True, True),
        ("Daniela Morales", 330, 70, "urgente", "traumatologia", False, False),
        ("Luciana Fuster", 340, 45, "leve", "cardiologia", False, False),
        ("Diego", 360, 180, "urgente", "cardiologia", True, False),
        ("Sebastián Flores", 360, 130, "emergencia", "cardiologia", True, True),
        ("Esteban Navarro", 375, 85, "urgente", "pediatria", False, True),
        ("Valeria Mazza", 395, 60, "leve", "traumatologia", False, False),
        ("Joaquín Salazar", 405, 90, "urgente", "pediatria", False, True),
        ("Sofia Franco", 420, 45, "leve", "obstetricia", False, False),
        ("Gabriela Paredes", 420, 60, "leve", "medicina general", False, False),
        ("Mateo Aguilar", 435, 120, "emergencia", "medicina general", True, True),
        ("Martín Cabrera", 450, 140, "emergencia", "traumatologia", True, True),
        ("Jorge Benavides", 470, 45, "leve", "cardiologia", False, False),
        ("Joaquin", 480, 90, "emergencia", "traumatologia", False, False),
        ("Fano Alonso", 495, 75, "urgente", "cirugia", False, True),
        ("Alejandro Ruiz", 495, 110, "urgente", "cirugia", False, True),
        ("Shirley Cherres", 540, 60, "urgente", "obstetricia", False, False),
        ("Cristian Navarro", 540, 160, "emergencia", "traumatologia", True, True),
        ("Isabella Torres", 570, 35, "urgente", "obstetricia", False, True),
        ("Daniela Darcourt", 585, 65, "urgente", "cardiologia", False, True),
        ("Alejandro", 600, 120, "leve", "traumatologia", True, False),
        ("Alejandra Baigorria", 645, 55, "leve", "cardiologia", False, False),
        ("Valentina", 660, 30, "urgente", "medicina general", False, True),
        ("Javier Molina", 675, 100, "urgente", "pediatria", False, True),
        ("Antonella Aservi", 705, 45, "leve", "traumatologia", False, False),
        ("Renato Tapia", 720, 100, "emergencia", "pediatria", True, False),
        ("Emilia Drago", 735, 160, "emergencia", "medicina general", True, True),
        ("Mauricio Mesones", 780, 45, "leve", "cardiologia", False, False),
        ("Facundo", 795, 115, "emergencia", "cirugia", True, True),
        ("Brunella Horna", 825, 50, "leve", "obstetricia", False, False),
        ("Fernanflo", 840, 150, "urgente", "traumatologia", True, True),
        ("Julieta Castro", 885, 60, "leve", "traumatologia", False, False),
        ("Mariana", 900, 60, "emergencia", "medicina general", False, True),
        ("Andres Palacios", 915, 130, "emergencia", "traumatologia", True, True),
        ("Milagros Soto", 945, 45, "leve", "cardiologia", False, False),
        ("Shirley Arica", 960, 90, "leve", "cardiologia", False, False),
        ("Benjamin Ponce", 975, 95, "urgente", "pediatria", False, True),
        ("Ariana Grande", 1005, 55, "leve", "obstetricia", False, False),
        ("Lolo Fernandez", 1020, 40, "urgente", "traumatologia", True, True),
        ("Franco Cabrera", 1035, 150, "emergencia", "medicina general", True, True),
        ("Michael J", 1080, 120, "urgente", "pediatria", True, True),
        ("Cristian C", 1110, 120, "emergencia", "cirugia", True, True),
        ("Bruno M", 1200, 75, "leve", "medicina general", False, False)
    ]
    for d in datos:
        hospital.registrar_paciente(*d)
    return len(datos)

def leer_numero(mensaje, minimo=0):
    while True:
        try:
            valor = int(input(mensaje))
            if valor < minimo:
                print(f"El valor mínimo es {minimo}.")
                continue
            return valor
        except ValueError:
            print("Ingresa un número entero válido")

def leer_booleano(mensaje):
    while True:
        respuesta = input(mensaje).strip().lower()
        if respuesta in ("s", "si", "sí"):
            return True
        if respuesta == "n":
            return False
        print("Ingresa S para Sí o N para No.")

def leer_prioridad():
    opciones = {"1": "emergencia", "2": "urgente", "3": "leve"}
    while True:
        entrada = input("* Prioridad => [1] Emergencia | [2] Urgente | [3] Leve : ").strip()
        if entrada in opciones:
            return opciones[entrada]
        print("Ingresa un número del 1 al 3.")

def leer_especialidad():#("cardiologia", "pediatria", "traumatologia", "medicina general", "obstetricia", "cirugia",)
    print("* Especialidad\n[1] Cardiología     [2] Pediatría\n[3] Traumatología   [4] Medicina general\n[5] Obstetricia     [6] Cirugía")
    opciones = {"1": "cardiologia", "2": "pediatria", "3": "traumatologia", "4": "medicina general", "5": "obstetricia", "6": "cirugia"}
    while True:
        entrada = input("Selecciona: ").strip()
        if entrada in opciones:
            return opciones[entrada]
        print("Opción inválida.")

def menu():
    hospital = Hospital(camas=14, diagnosticos=10, cantidad_consultorios=15)
    while True:
        print("\n" + "SISTENA DE COLA DE ATENCION HOSPITALARIA".center(80, "*") )
        print(f"| Médicos: {hospital.total_medicos} / Consultorios: {len(hospital.consultorios)} / Camas: {hospital.total_camas} / Diagnósticos: {hospital.total_diagnosticos}\n|")
        print("| [1] Registrar paciente\n| [2] Registrar médico\n| [3] Ejecutar simulación y comparar algoritmos\n| [4] Mostrar médicos, turnos y consultorios")
        print("| [5] Reiniciar simulación\n| [0] salir\n|")
        print("| Presione [D] para cargar datos de demostración y ejecutar")
        print("*"*80)
        opcion = input("Elige una opción: ").strip().lower()

        if opcion == "1":
            if hospital.ejecutado:
                print("La simulación ya fue procesada. Reiníciala antes de registrar más pacientes.")
            else:
                subprocess.run("cls", shell=True)
                print("REGISTRAR PACIENTE")
                nombre = input("* Nombre del paciente: ").strip()
                if not nombre:
                    print("El nombre no puede estar vacío.")
                    continue
                while True:
                    hora_llegada = input("* Hora de llegada del paciente (HH:MM, formato 24 horas): ").strip()
                    try:
                        llegada = hospital.convertir_hora_llegada(hora_llegada)
                        break
                    except ValueError as error:
                        print("Hora inválida:", error)
                duracion = leer_numero("* Duración estimada de atención (minutos): ", 1)
                prioridad = leer_prioridad()
                esp = leer_especialidad()
                cama = leer_booleano("* ¿Necesita cama? (S/N): ")
                diag = leer_booleano("* ¿Necesita diagnóstico? (S/N): ")
                try:
                    hospital.registrar_paciente(nombre, llegada, duracion, prioridad, esp, cama, diag)
                    print("¡Paciente registrado! :")
                    print(f"Nombre: {nombre} | Llegada: {hora_llegada} | Duracion: {duracion} minutos | Prioridad: {prioridad} | Especialidad: {esp} | Cama: {cama} | Diagnostico: {diag}")
                except ValueError as error:
                    print("Error:", error)

        elif opcion == "2":
            if hospital.ejecutado:
                print("Reinicia la simulación antes de modificar la plantilla médica.")
                continue
            subprocess.run("cls", shell=True)
            print("REGISTRAR MEDICO")
            nombre_medico = input("Nombre del médico: ").strip()
            especialidad_medico = leer_especialidad()
            hora_inicio = input("Hora de inicio (HH:MM): ").strip()
            hora_fin = input("Hora de fin (HH:MM): ").strip()
            try:
                hospital.registrar_medico(nombre_medico, especialidad_medico, hora_inicio, hora_fin)
            except ValueError as error:
                print("Error:", error)

        elif opcion == "3":
            if not hospital.pacientes:
                print("No hay pacientes registrados.")
                continue
            if hospital.ejecutado:
                print("Reinicia la simulación antes de ejecutarla nuevamente.")
                continue
            #comparación inicial con todos los pacientes que llegan al minuto cero
            subprocess.run("cls", shell=True)
            print("Simulacion")
            candidatos = [p for p in hospital.pacientes if p.Llegada == 0]
            disponibles, mapa = hospital.medicos_elegibles(candidatos)
            comparar_algoritmos(candidatos, mapa, hospital.total_camas, hospital.total_diagnosticos,min(len(hospital.consultorios), len(disponibles)), 0)

            print("\n" + "DEMOSTRACIÓN CON PROGRAMACIÓN DINÁMICA".center(100, "/"))
            hospital.simular(1)
            hospital.reiniciar()
            print("\n" + "DEMOSTRACIÓN CON DIVIDE Y VENCERÁS".center(100, "/"))
            hospital.simular(2)
            print(f"\n{"DEMOSTRACION FINALIZADA".center(120,"*")}\n")

        elif opcion == "4":
            subprocess.run("cls", shell=True)
            hospital.mostrar_medicos()
            print("\nCONSULTORIOS:")
            for c in hospital.consultorios:
                estado = f"ocupado por médico #{c.ocupado_por}" if c.ocupado_por else "libre"
                print(f"Consultorio {c.Id}: {estado}")

        elif opcion == "5":
            hospital.reiniciar()

        elif opcion == "d" or opcion == "D":
            subprocess.run("cls", shell=True)
            hospitalDemostracion = Hospital(camas=12, diagnosticos=10, cantidad_consultorios=4)
            cantidad_pacientes = datos_demostracion(hospitalDemostracion)
            print(f"\n{"DEMOSTRACION INICIADA".center(120,"*")}\n" + 
                  f"Medicos: {hospitalDemostracion.total_medicos} | Pacientes llegados: {cantidad_pacientes}| Camas: 10 | Diagnostiocos: 10 | Consultorios: 4")
            disponibles, mapa = hospitalDemostracion.medicos_elegibles(hospitalDemostracion.pacientes)
            comparar_algoritmos(hospitalDemostracion.pacientes, mapa, hospitalDemostracion.total_camas, hospitalDemostracion.total_diagnosticos,
                                min(len(hospitalDemostracion.consultorios), len(disponibles)), 0)
            print("\n" + "DEMOSTRACIÓN CON PROGRAMACIÓN DINÁMICA".center(100, "/"))
            hospitalDemostracion.simular(1)
            hospitalDemostracion.reiniciar()
            print("\n" + "DEMOSTRACIÓN CON DIVIDE Y VENCERÁS".center(100, "/"))
            hospitalDemostracion.simular(2)
            print(f"\n{"DEMOSTRACION FINALIZADA".center(120,"*")}\n")

        elif opcion == "0":
            print("Saliendo...")
            input()
            subprocess.run("cls", shell=True)
            break
        else:
            print("Opción inválida.")
        
        input("\nPresione cualquie tecla para continuar: ")
        subprocess.run("cls", shell=True)

if __name__ == "__main__":
    menu()